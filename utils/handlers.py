import streamlit as st
import time
import os
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from .text_utils import clean_text, extract_hs_codes
from .hs_manual_utils import (
    get_tariff_info_for_codes,
    get_manual_info_for_codes,
    prepare_general_rules,
    analyze_user_provided_codes
)
from .search_engines import ParallelHSSearcher
from .query_expander import QueryExpander
from .api_retry import retry_on_api_error, retry_api_call, extract_retry_delay_from_error

# API client는 main.py에서 파라미터로 전달받음

# prompts.py에서 프롬프트 import
from prompts import DOMESTIC_CONTEXT, OVERSEAS_CONTEXT


# ==================== 유틸리티 함수 ====================

def highlight_keywords(text, keywords):
    """텍스트에서 키워드를 형광색으로 하이라이트 (토큰 기반)"""
    if not text or not keywords:
        return text

    # 키워드가 문자열이면 공백으로 분리하여 토큰화
    if isinstance(keywords, str):
        # 특수문자 제거 및 공백 기준 분리
        keywords = re.sub(r'[^\w\s]', ' ', keywords).split()
        # 길이 2 이상인 토큰만 사용
        keywords = [kw.strip() for kw in keywords if len(kw.strip()) >= 2]

    if not keywords:
        return text

    result = text
    # 각 토큰을 개별적으로 하이라이트
    for keyword in keywords:
        if not keyword or len(keyword.strip()) < 2:  # 너무 짧은 키워드는 스킵
            continue
        # 대소문자 구분 없이 치환 (re.IGNORECASE)
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        result = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', result)

    return result


# ==================== Multi-Agent 공통 로직 ====================

def truncate_text_at_sentence(text, max_chars):
    """텍스트를 문장 단위로 제한"""
    if not text or len(text) <= max_chars:
        return text

    # 일단 max_chars까지 자르기
    truncated = text[:max_chars]

    # 문장 끝 찾기 (마침표, 한글 마침표, 줄바꿈)
    last_sentence_end = max(
        truncated.rfind('.'),
        truncated.rfind('。'),
        truncated.rfind('\n')
    )

    # 80% 이상 지점에서 문장 끝을 찾았으면 그곳에서 자르기
    if last_sentence_end > max_chars * 0.8:
        return truncated[:last_sentence_end + 1]

    # 못 찾았으면 그냥 자르고 ... 추가
    return truncated + "..."


def truncate_case_text(case, max_chars=1500):
    """사례의 긴 텍스트 필드를 제한 (문장 단위)"""
    truncated_case = case.copy()

    # description 필드 제한
    if 'description' in truncated_case and truncated_case['description']:
        truncated_case['description'] = truncate_text_at_sentence(
            truncated_case['description'], max_chars
        )

    # reply 필드 제한 (해외 사례)
    if 'reply' in truncated_case and truncated_case['reply']:
        truncated_case['reply'] = truncate_text_at_sentence(
            truncated_case['reply'], max_chars
        )

    # decision_reason 필드 제한 (국내 사례)
    if 'decision_reason' in truncated_case and truncated_case['decision_reason']:
        truncated_case['decision_reason'] = truncate_text_at_sentence(
            truncated_case['decision_reason'], max_chars
        )

    return truncated_case


def _process_single_group(group_id, group_cases, context_prompt, user_input, analysis_type, client):
    """단일 그룹 처리 함수 (병렬 실행용, 3단계 Fallback 적용)"""
    try:
        # 텍스트 길이 제한 적용 (토큰 소비 감소)
        truncated_cases = [truncate_case_text(case, max_chars=1500) for case in group_cases]

        # 그룹 데이터를 컨텍스트로 변환
        source_label = "국내 관세청" if analysis_type == 'domestic' else "해외 관세청"
        relevant = "\n\n".join([
            f"출처: {source_label}\n항목: {json.dumps(case, ensure_ascii=False)}"
            for case in truncated_cases
        ])

        prompt = f"{context_prompt}\n\n관련 데이터 ({source_label}, 그룹{group_id+1}):\n{relevant}\n\n사용자: {user_input}\n"

        start_time = datetime.now()

        # Fallback 모델 리스트 (우선순위 순)
        # 1. Primary: Gemini 2.5 Flash (기본)
        # 2. Fallback 1: Gemini 2.5 Flash Lite (고속/경량)
        # 3. Fallback 2: Gemini 2.0 Flash (안정성)
        fallback_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash"
        ]
        
        last_error = None
        answer = None
        
        for model_name in fallback_models:
            try:
                # 단일 모델 재시도 없이 1회 시도 (Fallback으로 빠른 전환 유도)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                answer = clean_text(response.text)
                
                # 성공 시 루프 탈출
                if answer:
                    break
                    
            except Exception as e:
                # 429 에러인 경우 대기 후 다음 모델로 전환
                if isinstance(e, APIError) and e.code == 429:
                    suggested_wait = extract_retry_delay_from_error(e) or 0
                    # 기본 4초와 API 제안 시간 중 더 긴 시간 대기
                    wait_time = max(4.0, suggested_wait)
                    time.sleep(wait_time)
                
                # 현재 모델 실패, 에러 기록하고 다음 모델 시도
                last_error = e
                # print(f"Group {group_id+1} failed with {model_name}: {str(e)}") # 디버그용
                continue

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        if answer:
            return group_id, answer, start_time, processing_time
        else:
            # 모든 모델 실패
            raise last_error if last_error else Exception("All models failed")

    except APIError as e:
        # API 에러 (모든 Fallback 실패 후)
        error_msg = f"그룹 {group_id+1} API 오류 ({e.code}): {e.message}"
        return group_id, error_msg, datetime.now(), 0.0

    except Exception as e:
        # 기타 예외
        error_msg = f"그룹 {group_id+1} 분석 중 오류 발생: {str(e)}"
        return group_id, error_msg, datetime.now(), 0.0


def _run_group_parallel_analysis(groups, context_prompt, user_input, analysis_type, client, ui_container=None):
    """5개 그룹을 병렬로 분석하는 공통 함수"""

    # UI 초기화
    if ui_container:
        progress_bar = ui_container.progress(0, text="병렬 AI 분석 시작...")
        responses_container = ui_container.container()

    # 병렬 처리
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        # 그룹 시작 시 순차 딜레이 적용 (동시 API 호출 충돌 방지)
        futures = []
        for i in range(5):
            # 각 그룹마다 0.3초씩 지연 시작
            if i > 0:
                time.sleep(0.3)
            futures.append(
                executor.submit(_process_single_group, i, groups[i], context_prompt, user_input, analysis_type, client)
            )

        for future in as_completed(futures):
            group_id, answer, start_time, processing_time = future.result()
            results[group_id] = answer

            # session_state에 결과 저장
            if ui_container:
                analysis_result = {
                    'type': analysis_type,
                    'group_id': group_id,
                    'answer': answer,
                    'start_time': start_time.strftime('%H:%M:%S'),
                    'processing_time': processing_time
                }
                st.session_state.ai_analysis_results.append(analysis_result)

                # 실시간 UI 업데이트
                with responses_container:
                    emoji = "🤖" if analysis_type == 'domestic' else "🌐"
                    st.success(f"{emoji} **그룹 {group_id+1} AI 분석 완료** ({processing_time:.1f}초)")
                    with st.container():
                        st.write(f"⏰ {start_time.strftime('%H:%M:%S')}")
                        st.markdown("**분석 결과:**")
                        st.info(answer)
                        st.divider()

                progress_bar.progress(len(results)/5, text=f"완료: {len(results)}/5 그룹")

    # 순서대로 정렬
    group_answers = [results[i] for i in range(5)]
    return group_answers


def _run_head_agent(group_answers, context_prompt, user_input, analysis_type, client, ui_container=None):
    """Head Agent가 5개 답변을 종합하는 함수 (3단계 Fallback 적용)"""

    if ui_container:
        ui_container.progress(1.0, text="Head AI 최종 분석 중...")
        ui_container.info("🧠 **Head AI가 모든 분석을 종합하는 중...**")

    final_answer = ""
    analysis_label = "국내 HS 분류 사례" if analysis_type == 'domestic' else "해외 HS 분류 사례"
    head_prompt = f"{context_prompt}\n\n아래는 {analysis_label} 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"

    for idx, ans in enumerate(group_answers):
        head_prompt += f"[그룹{idx+1} 답변]\n{ans}\n\n"
    head_prompt += f"\n사용자: {user_input}\n"

    # Fallback 모델 리스트 (우선순위 순)
    # 1. Primary: Gemini 3.0 Flash Preview (최신)
    # 2. Fallback 1: Gemini 2.5 Flash Lite (고속/경량)
    # 3. Fallback 2: Gemini 2.0 Flash (안정성)
    fallback_models = [
        "gemini-3-flash-preview", 
        "gemini-2.5-flash-lite", 
        "gemini-2.0-flash"
    ]
    
    last_error = None

    for model_name in fallback_models:
        try:
            # 각 모델별 시도 (재시도 데코레이터 없이 1회 시도, 빠른 전환을 위해)
            # 필요하다면 여기서도 재시도를 넣을 수 있으나, Fallback 전략상 즉시 다음 모델로 넘기는 것이 유리함
            
            if ui_container:
                # 현재 시도 중인 모델 표시 (디버그용 정보)
                # ui_container.caption(f"Trying model: {model_name}...") 
                pass

            head_response = client.models.generate_content(
                model=model_name,
                contents=head_prompt
            )
            final_answer = clean_text(head_response.text)
            
            # 성공 시 루프 탈출
            if final_answer:
                break
                
        except Exception as e:
            last_error = e
            
            # 429 에러인 경우 대기 후 다음 모델로 전환
            if isinstance(e, APIError) and e.code == 429:
                s_wait = extract_retry_delay_from_error(e) or 0
                w_time = max(4.0, s_wait)
                if ui_container:
                     ui_container.warning(f"⚠️ 모델({model_name}) 사용량 초과. {w_time:.1f}초 대기 후 다음 모델로 전환합니다.")
                time.sleep(w_time)
            
            # 현재 모델 실패, 다음 모델 시도
            if ui_container:
                print(f"Model {model_name} failed: {str(e)}") # 서버 로그용
            continue

    # 모든 모델 실패 시 처리
    if not final_answer:
        error_msg = str(last_error) if last_error else "Unknown Error"
        final_answer = f"Head AI 분석 중 오류가 발생했습니다 (All models failed). 마지막 오류: {error_msg}\n\n그룹별 분석 결과를 참고해주세요."
        if ui_container:
            ui_container.error(f"⚠️ Head AI 오류: 모든 모델 시도 실패. ({error_msg})")

    if ui_container and final_answer and not final_answer.startswith("Head AI 분석 중 오류"):
        ui_container.progress(1.0, text="분석 완료!")
        ui_container.success("✅ **모든 AI 분석이 완료되었습니다**")
        ui_container.info("📋 **패널을 접고 아래에서 최종 답변을 확인하세요**")

    return final_answer


# ==================== 통합 Multi-Agent 핸들러 ====================

def handle_multi_agent_analysis(user_input, context, hs_manager, analysis_type, client, ui_container=None):
    """
    통합 Multi-Agent 분석 핸들러 (쿼리 확장 적용)

    Args:
        user_input: 사용자 질문
        context: 대화 컨텍스트 (현재 미사용)
        hs_manager: HS 데이터 매니저
        analysis_type: 'domestic' 또는 'overseas'
        client: Google Gemini API client
        ui_container: Streamlit UI 컨테이너 (optional)

    Returns:
        최종 분석 결과 문자열
    """

    # 분석 타입에 따라 프롬프트 및 검색 엔진 선택
    if analysis_type == 'domestic':
        context_prompt = DOMESTIC_CONTEXT
        search_func = hs_manager.search_domestic_tfidf
        ui_message = "🔍 **국내 HS 분류사례 분석 시작**"
    elif analysis_type == 'overseas':
        context_prompt = OVERSEAS_CONTEXT
        search_func = hs_manager.search_overseas_tfidf
        ui_message = "🌍 **해외 HS 분류사례 분석 시작**"
    else:
        raise ValueError(f"Invalid analysis_type: {analysis_type}. Must be 'domestic' or 'overseas'.")

    # UI 초기화
    if ui_container:
        with ui_container:
            st.info(ui_message)

    # 쿼리 확장 단계 추가
    try:
        if ui_container:
            with ui_container:
                st.info("🧠 **AI 쿼리 확장 중...**")

        expander = QueryExpander(client, 'balanced')
        expansion_result = expander.expand_query(user_input)

        # session_state에 쿼리 확장 결과 저장
        if ui_container and expansion_result['expansion_applied']:
            if 'query_expansion_result' not in st.session_state:
                st.session_state.query_expansion_result = None

            st.session_state.query_expansion_result = {
                'target_product': expansion_result.get('target_product', 'N/A'),
                'material': expansion_result.get('material', ''),
                'components': expansion_result.get('components', ''),
                'function': expansion_result.get('function', ''),
                'keyword_groups': expansion_result.get('keyword_groups', {}),
                'expanded_query': expansion_result.get('expanded_query', ''),
                'original_query': user_input
            }

            with ui_container:
                st.success(f"✅ **쿼리 확장 완료**: {expansion_result['target_product']}")
                st.text(f"확장된 쿼리: {expansion_result['expanded_query'][:100]}...")

        # 확장된 쿼리로 검색
        expanded_query = expansion_result['expanded_query']

    except Exception as e:
        if ui_container:
            st.session_state.query_expansion_result = None
            with ui_container:
                st.warning(f"⚠️ 쿼리 확장 실패, 원본 쿼리 사용: {str(e)}")
        expanded_query = user_input

    # TF-IDF 기반 검색으로 상위 100개 사례 추출 (확장된 쿼리 사용)
    top_cases = search_func(expanded_query, top_k=100, min_similarity=0.05)

    # 5개 그룹으로 분할 (각 그룹 20개)
    group_size = len(top_cases) // 5
    groups = [top_cases[i*group_size:(i+1)*group_size if i < 4 else len(top_cases)] for i in range(5)]

    # 5개 그룹 병렬 분석
    group_answers = _run_group_parallel_analysis(groups, context_prompt, user_input, analysis_type, client, ui_container)

    # Head Agent 최종 종합
    final_answer = _run_head_agent(group_answers, context_prompt, user_input, analysis_type, client, ui_container)

    return final_answer


# ==================== 기존 인터페이스 유지 (래퍼 함수) ====================

def handle_hs_classification_cases(user_input, context, hs_manager, client, ui_container=None):
    """국내 HS 분류 사례 처리 (래퍼 함수)"""
    return handle_multi_agent_analysis(user_input, context, hs_manager, 'domestic', client, ui_container)


def handle_overseas_hs(user_input, context, hs_manager, client, ui_container=None):
    """해외 HS 분류 사례 처리 (래퍼 함수)"""
    return handle_multi_agent_analysis(user_input, context, hs_manager, 'overseas', client, ui_container)


# ==================== 기타 핸들러 함수 ====================

def handle_web_search(user_input, context, hs_manager, client):
    """웹 검색 처리 함수 (재시도 로직 포함)"""
    web_context = """당신은 HS 품목분류 전문가입니다.

사용자의 질문에 대해 최신 웹 정보를 검색하여 물품개요, 용도, 기술개발, 산업동향 등의 정보를 제공해주세요.
"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    prompt = f"{web_context}\n\n사용자: {user_input}\n"

    # 재시도 로직 적용
    @retry_on_api_error(max_retries=3, initial_delay=0.5)
    def _web_search_api_call():
        return client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config
        )

    response = _web_search_api_call()
    return clean_text(response.text)


def handle_hs_manual_with_user_codes(user_input, context, hs_manager, logger, extracted_codes, client, ui_container=None):
    """사용자 제시 HS코드 기반 해설서 분석"""

    # UI 컨테이너가 제공된 경우 분석 과정 표시
    if ui_container:
        with ui_container:
            st.info("🔍 **사용자 제시 HS코드 분석 시작**")
            progress_bar = st.progress(0, text="HS코드 분석 중...")
            analysis_container = st.container()

    logger.log_actual("SUCCESS", f"Found {len(extracted_codes)} HS codes", f"{', '.join(extracted_codes)}")

    if ui_container:
        progress_bar.progress(0.2, text=f"{len(extracted_codes)}개 HS코드 발견...")
        with analysis_container:
            st.success(f"✅ **{len(extracted_codes)}개 HS코드 발견**: {', '.join(extracted_codes)}")

    # 2단계: 각 HS코드별 품목분류표 정보 수집
    logger.log_actual("INFO", "Collecting tariff table information...")
    tariff_info = get_tariff_info_for_codes(extracted_codes)

    if ui_container:
        progress_bar.progress(0.4, text="품목분류표 정보 수집 중...")

    # 3단계: 각 HS코드별 해설서 정보 수집 및 요약
    logger.log_actual("INFO", "Collecting and summarizing manual information...")
    manual_info = get_manual_info_for_codes(extracted_codes, logger, client)

    if ui_container:
        progress_bar.progress(0.6, text="해설서 정보 수집 및 요약 중...")

        # 수집된 정보 표시
        with analysis_container:
            st.markdown("### 📊 **HS코드별 상세 정보**")

            for code in extracted_codes:
                st.markdown(f"#### 🔢 **HS코드: {code}**")

                col1, col2 = st.columns([1, 1])
                with col1:
                    if code in tariff_info:
                        st.write(f"**📋 국문품명**: {tariff_info[code].get('korean_name', 'N/A')}")
                        st.write(f"**📋 영문품명**: {tariff_info[code].get('english_name', 'N/A')}")

                with col2:
                    if code in manual_info:
                        st.write(f"**📚 해설서**: 수집 완료")
                        if manual_info[code].get('summary_used'):
                            st.write(f"**🤖 요약**: 적용됨")

                st.divider()

    # 4단계: 통칙 준비
    logger.log_actual("INFO", "Preparing general rules...")
    general_rules = prepare_general_rules()

    if ui_container:
        progress_bar.progress(0.8, text="최종 AI 분석 준비 중...")

    # 5단계: 최종 AI 분석
    logger.log_actual("AI", "Starting final AI analysis...")
    final_answer = analyze_user_provided_codes(user_input, extracted_codes, tariff_info, manual_info, general_rules, context, client)

    if ui_container:
        progress_bar.progress(1.0, text="분석 완료!")
        st.success("🧠 **AI 전문가 분석이 완료되었습니다**")
        st.info("📋 **아래에서 최종 답변을 확인하세요**")

    logger.log_actual("SUCCESS", "User-provided codes analysis completed", f"{len(final_answer)} chars")
    return final_answer


def handle_domestic_case_lookup(user_input, hs_manager):
    """국내 분류사례 원문 검색 처리 함수"""

    # 1. 참고문서번호 직접 검색
    ref_pattern = r'품목분류\d+과-\d+'
    match = re.search(ref_pattern, user_input)

    if match:
        ref_id = match.group()
        case = hs_manager.find_domestic_case_by_id(ref_id)
        if case:
            # 참고문서번호 유효성 검증 (데이터 오류 필터링)
            if case.get('reference_id') and case['reference_id'] != '-1':
                return format_domestic_case_detail(case, query=ref_id)
            else:
                return f"⚠️ 참고문서번호 '{ref_id}'의 데이터에 문제가 있습니다.\n\n키워드 검색을 시도해주세요."
        else:
            return f"⚠️ 참고문서번호 '{ref_id}'에 해당하는 사례를 찾을 수 없습니다.\n\n다른 문서번호나 키워드로 다시 검색해주세요."

    # 2. 키워드 기반 단순 문자열 검색
    results = hs_manager.search_domestic_by_keyword(user_input, top_k=10)

    if not results:
        return f"""⚠️ **"{user_input}"에 대한 검색 결과가 없습니다**

**가능한 원인:**
- 해당 키워드가 포함된 분류사례가 데이터에 없습니다
- 검색어가 원문에 정확히 일치하지 않습니다

**검색 팁:**
- 품목명의 핵심 키워드 사용 (예: '섬유유연제', '폴리아미드')
- 영문 품목명 시도 (예: 'softening', 'polyamide')
- 더 짧고 일반적인 단어 사용 (예: '머그컵' → '컵', 'mug')
- 띄어쓰기 변경 시도 (예: '폴리아미드호스' → '폴리아미드 호스')

**다른 검색 방법:**
- **국내 분류사례 기반 HS 추천**: AI가 유사 사례를 분석하여 HS코드 추천 (TF-IDF 사용)
- **웹 검색**: 최신 정보 및 일반 품목 정보 검색"""

    return format_domestic_case_list(results, query=user_input)


def format_domestic_case_detail(case, query=None):
    """국내 사례 상세 포맷"""
    # 키워드 하이라이트 적용
    product_name = highlight_keywords(case.get('product_name', 'N/A'), query) if query else case.get('product_name', 'N/A')
    description = highlight_keywords(case.get('description', 'N/A'), query) if query else case.get('description', 'N/A')
    decision_reason = highlight_keywords(case.get('decision_reason', 'N/A'), query) if query else case.get('decision_reason', 'N/A')

    return f"""---
<div class="case-detail">

## 📋 국내 분류사례 상세 정보

<div class="info-table">

| 항목 | 내용 |
|------|------|
| 📄 **참고문서번호** | {case.get('reference_id', 'N/A')} |
| 📅 **결정일자** | {case.get('decision_date', 'N/A')} |
| 🏛️ **결정기관** | {case.get('organization', 'N/A')} |
| 🔢 **HS 코드** | {case.get('hs_code', 'N/A')} |

</div>

---

### 📦 품목명
{product_name}

---

### 📝 품목 설명
{description}

---

### ⚖️ 분류 근거
{decision_reason}

</div>
"""


def format_domestic_case_list(results, query):
    """국내 사례 목록 포맷 (Expander 방식)"""
    output = f"## 🔍 \"{query}\" 검색 결과 ({len(results)}건)\n\n"

    for idx, case in enumerate(results, 1):
        product_name = case.get('product_name', 'N/A')
        ref_id = case.get('reference_id', 'N/A')
        hs_code = case.get('hs_code', 'N/A')
        decision_date = case.get('decision_date', 'N/A')

        # 품목명이 너무 길면 자르기 (Expander 제목용)
        product_name_display = product_name[:60] + "..." if len(product_name) > 60 else product_name
        # 제목에도 하이라이트 적용
        product_name_display = highlight_keywords(product_name_display, query)

        # 카드형 Expander
        output += f"""<div class="case-card domestic">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="product-name">{product_name_display}</span>
<span class="date">{decision_date}</span>
</summary>

<div class="case-content">
"""

        # Expander 내용 (전체 상세 정보, 하이라이트 적용)
        output += format_domestic_case_detail(case, query=query)

        output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output


def handle_overseas_case_lookup(user_input, hs_manager):
    """해외 분류사례 원문 검색 처리 함수"""

    # 1. 참고문서번호 검색 (미국/EU 패턴)
    us_pattern = r'(NY|HQ|LA|SF|N)\s+[A-Z]?\d+'
    match = re.search(us_pattern, user_input, re.IGNORECASE)

    if match:
        ref_id = match.group()
        result = hs_manager.find_overseas_case_by_id(ref_id)
        if result:
            return format_overseas_case_detail(result['case'], result['country'], query=ref_id)
        else:
            return f"⚠️ 참고문서번호 '{ref_id}'에 해당하는 사례를 찾을 수 없습니다.\n\n다른 문서번호나 키워드로 다시 검색해주세요."

    # 2. HS 코드 검색
    hs_pattern = r'\b\d{4}(\.\d{2}){0,2}\b'
    match = re.search(hs_pattern, user_input)

    if match:
        hs_code = match.group()
        results = hs_manager.search_overseas_by_hs_code(hs_code, top_k=10)
        if results:
            return format_overseas_case_list_by_hs(results, hs_code)

    # 3. 키워드 기반 단순 문자열 검색
    results = hs_manager.search_overseas_by_keyword(user_input, top_k=10)

    if not results:
        return f"""⚠️ **"{user_input}"에 대한 검색 결과가 없습니다**

**가능한 원인:**
- 해당 키워드가 포함된 분류사례가 데이터에 없습니다
- 검색어가 원문에 정확히 일치하지 않습니다

**검색 팁:**
- 영문 품목명 사용 (예: 'fabric', 'textile', 'bag')
- 더 짧고 일반적인 단어 사용 (예: 'ceramic mug' → 'mug', 'ceramic')
- 띄어쓰기 변경 시도
- HS 코드로 검색 (예: '5515.12', '4202.92')
- 참고문서번호로 검색 (예: 'NY N338825')

**다른 검색 방법:**
- **해외 분류사례 기반 HS 추천**: AI가 유사 사례를 분석하여 HS코드 추천 (TF-IDF 사용)
- **웹 검색**: 최신 정보 및 일반 품목 정보 검색"""

    # 결과를 국가별로 분리
    us_results = []
    eu_results = []

    for item in results:
        # 원본 데이터에서 국가 판단
        if 'hs_classification_data_us' in str(item) or item.get('organization', '').startswith('New York'):
            us_results.append(item)
        else:
            eu_results.append(item)

    return format_overseas_case_list(us_results, eu_results, query=user_input)


def format_overseas_case_detail(case, country, query=None):
    """해외 사례 상세 포맷"""
    country_flag = "🇺🇸" if country == "US" else "🇪🇺"
    country_name = "미국 CBP" if country == "US" else "EU 관세청"

    # 키워드 하이라이트 적용
    reply = highlight_keywords(case.get('reply', 'N/A'), query) if query else case.get('reply', 'N/A')
    description = highlight_keywords(case.get('description', 'N/A'), query) if query else case.get('description', 'N/A')

    return f"""---
<div class="case-detail">

## {country_flag} {country_name} 분류사례 상세 정보

<div class="info-table">

| 항목 | 내용 |
|------|------|
| 📄 **참고문서번호** | {case.get('reference_id', 'N/A')} |
| 📅 **결정일자** | {case.get('decision_date', 'N/A')} |
| 🏛️ **결정기관** | {case.get('organization', 'N/A')} |
| 🔢 **HS 코드** | {case.get('hs_code', 'N/A')} |
| 📆 **연도** | {case.get('year', 'N/A')} |

</div>

---

### 📋 요약
{reply}

---

### 📝 상세 내용
{description}

</div>
"""


def format_overseas_case_list_by_hs(results, hs_code):
    """HS 코드 기반 해외 사례 목록 포맷 (Expander 방식)"""
    output = f"## 🔍 HS 코드 \"{hs_code}\" 검색 결과 ({len(results)}건)\n\n"

    us_count = sum(1 for r in results if r['country'] == 'US')
    eu_count = len(results) - us_count

    output += f"- 🇺🇸 미국: {us_count}건\n"
    output += f"- 🇪🇺 EU: {eu_count}건\n\n"

    for idx, item in enumerate(results, 1):
        case = item['case']
        country = item['country']
        flag = "🇺🇸" if country == "US" else "🇪🇺"
        card_class = "us" if country == "US" else "eu"

        reply = case.get('reply', 'N/A')
        reply_short = reply[:80] + "..." if len(reply) > 80 else reply
        # 요약에도 하이라이트 적용
        reply_short = highlight_keywords(reply_short, hs_code)

        ref_id = case.get('reference_id', 'N/A')
        hs_code_display = case.get('hs_code', 'N/A')

        # 카드형 Expander
        output += f"""<div class="case-card {card_class}">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위 {flag}</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code_display}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

        # Expander 내용 (전체 상세 정보, 하이라이트 적용)
        output += format_overseas_case_detail(case, country, query=hs_code)

        output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output


def format_overseas_case_list(us_results, eu_results, query):
    """키워드 기반 해외 사례 목록 포맷 (국가별 구분, Expander 방식)"""
    total_count = len(us_results) + len(eu_results)
    output = f"## 🔍 \"{query}\" 검색 결과 ({total_count}건)\n\n"

    if us_results:
        output += f"### 🇺🇸 미국 ({len(us_results)}건)\n\n"
        for idx, case in enumerate(us_results, 1):
            reply = case.get('reply', 'N/A')
            reply_short = reply[:60] + "..." if len(reply) > 60 else reply
            # 요약에도 하이라이트 적용
            reply_short = highlight_keywords(reply_short, query)

            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # 카드형 Expander
            output += f"""<div class="case-card us">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

            # Expander 내용 (하이라이트 적용)
            output += format_overseas_case_detail(case, 'US', query=query)

            output += """</div>
</details>
</div>

"""

    if eu_results:
        output += f"\n---\n\n### 🇪🇺 EU ({len(eu_results)}건)\n\n"
        for idx, case in enumerate(eu_results, 1):
            reply = case.get('reply', 'N/A')
            reply_short = reply[:60] + "..." if len(reply) > 60 else reply
            # 요약에도 하이라이트 적용
            reply_short = highlight_keywords(reply_short, query)

            ref_id = case.get('reference_id', 'N/A')
            hs_code = case.get('hs_code', 'N/A')

            # 카드형 Expander
            output += f"""<div class="case-card eu">
<details>
<summary class="case-summary">
<span class="arrow">▶</span>
<span class="rank">{idx}위</span>
<span class="ref-id">{ref_id}</span>
<span class="hs-code">HS {hs_code}</span>
<span class="reply-preview">{reply_short}</span>
</summary>

<div class="case-content">
"""

            # Expander 내용 (하이라이트 적용)
            output += format_overseas_case_detail(case, 'EU', query=query)

            output += """</div>
</details>
</div>

"""

    output += "\n💡 **각 항목을 클릭하면 상세 정보를 확인할 수 있습니다.**"
    return output
