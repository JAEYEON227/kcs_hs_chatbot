import streamlit as st
from google import genai
from google.genai.errors import APIError
import time
from datetime import datetime

import os
from dotenv import load_dotenv
from utils import HSDataManager, extract_hs_codes, clean_text
from utils import handle_web_search, handle_hs_classification_cases, handle_overseas_hs, get_hs_explanations, handle_hs_manual_with_user_codes
from utils import handle_domestic_case_lookup, handle_overseas_case_lookup
from prompts import SYSTEM_PROMPT
from config import CATEGORY_MAPPING, LOGGER_ICONS, EXAMPLE_QUESTIONS

# Streamlit 페이지 설정
st.set_page_config(
    page_title="HS 품목분류 챗봇 (API Key 입력)",  # 브라우저 탭 제목
    page_icon="🔑",  # 브라우저 탭 아이콘
    layout="wide"  # 페이지 레이아웃을 넓게 설정
)

# 사용자 정의 CSS 스타일 추가
st.markdown("""
<style>
.stTextInput input {
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 16px;
}
/* 라디오 버튼 크기 및 글자 크기 증가 */
.stRadio > label {
    font-size: 16px !important;
    font-weight: 500 !important;
}
.stRadio > div {
    gap: 4px !important;
}
.stRadio > div > label {
    font-size: 17px !important;
    padding: 4px 0px !important;
}
.stRadio > div > label > div:first-child {
    margin-right: 8px !important;
}

/* 키워드 하이라이트 (노란 형광색) */
mark {
    background-color: #ffeb3b;
    color: #000;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}

/* 카드형 검색 결과 레이아웃 */
.case-card {
    margin: 12px 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    border: 2px solid transparent;
}

.case-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* 국내/해외 색상 구분 */
.case-card.domestic {
    border-left: 5px solid #1976d2;
    background-color: #f5f9ff;
}

.case-card.us {
    border-left: 5px solid #d32f2f;
    background-color: #fff5f5;
}

.case-card.eu {
    border-left: 5px solid #1565c0;
    background-color: #f0f7ff;
}

/* Expander 제목 스타일 */
.case-summary {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    cursor: pointer;
    background-color: white;
    font-size: 15px;
    font-weight: 500;
    transition: background-color 0.2s;
    list-style: none;
}

.case-summary::-webkit-details-marker {
    display: none;
}

.case-summary:hover {
    background-color: #fafafa;
}

details[open] .case-summary {
    border-bottom: 1px solid #e0e0e0;
    background-color: #f8f8f8;
}

/* 화살표 아이콘 */
.arrow {
    font-size: 14px;
    color: #666;
    transition: transform 0.3s;
    min-width: 16px;
}

details[open] .arrow {
    transform: rotate(90deg);
}

/* 순위 배지 */
.rank {
    background-color: #ff9800;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    min-width: 40px;
    text-align: center;
}

/* 참고문서번호 */
.ref-id {
    color: #1976d2;
    font-weight: 600;
    font-size: 14px;
    min-width: 140px;
}

/* HS 코드 */
.hs-code {
    background-color: #e3f2fd;
    color: #1565c0;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}

/* 품목명/요약 미리보기 */
.product-name, .reply-preview {
    flex: 1;
    color: #424242;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* 날짜 */
.date {
    color: #757575;
    font-size: 13px;
    white-space: nowrap;
}

/* Expander 내용 영역 */
.case-content {
    padding: 20px;
    background-color: #fafafa;
    line-height: 1.7;
}

/* 상세 정보 테이블 */
.info-table table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    background-color: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.info-table th, .info-table td {
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
}

.info-table th {
    background-color: #f5f5f5;
    font-weight: 600;
    color: #424242;
    width: 30%;
}

.info-table tr:last-child td {
    border-bottom: none;
}

/* 상세 정보 섹션 */
.case-detail h3 {
    color: #1976d2;
    margin-top: 20px;
    margin-bottom: 12px;
    font-size: 18px;
}

.case-detail h2 {
    color: #1565c0;
    margin-bottom: 16px;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .case-summary {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
        padding: 12px 16px;
    }

    .rank, .ref-id, .hs-code {
        min-width: auto;
    }

    .product-name, .reply-preview {
        white-space: normal;
    }
}
</style>
""", unsafe_allow_html=True)

# HS 데이터 매니저 초기화 (캐싱을 통해 성능 최적화)
@st.cache_resource(show_spinner=False)
def get_hs_manager():
    import os
    if os.path.exists('tfidf_indexes.pkl.gz') or os.path.exists('tfidf_indexes.pkl'):
        with st.spinner("📂 TF-IDF 인덱스 로딩 중... (1초 이내)"):
            return HSDataManager()
    else:
        with st.spinner("🔧 TF-IDF 검색 인덱스 구축 중... (최초 1회, 5-15초 소요)"):
            return HSDataManager()

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # 채팅 기록 저장

if 'selected_category' not in st.session_state:
    st.session_state.selected_category = "국내 분류사례 기반 HS 추천"  # 기본값

if 'context' not in st.session_state:
    # 초기 컨텍스트 설정
    st.session_state.context = SYSTEM_PROMPT

if 'ai_analysis_results' not in st.session_state:
    st.session_state.ai_analysis_results = []

class RealTimeProcessLogger:
    def __init__(self, container):
        self.container = container
        self.log_placeholder = container.empty()
        self.logs = []
        self.start_time = time.time()
    
    def log_actual(self, level, message, data=None):
        """실제 진행 상황만 기록"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        log_entry = {
            "time": timestamp,
            "elapsed": f"{elapsed:.2f}s",
            "level": level,
            "message": message,
            "data": data
        }
        self.logs.append(log_entry)
        self.update_display()
    
    def update_display(self):
        log_text = ""
        icons = LOGGER_ICONS

        for log in self.logs[-8:]:
            icon = icons.get(log['level'], "📝")
            data_str = f" | {log['data']}" if log['data'] else ""
            log_text += f"`{log['time']}` `+{log['elapsed']}` {icon} {log['message']}{data_str}\n\n"
        
        self.log_placeholder.markdown(log_text)
    
    def clear(self):
        self.logs = []
        self.log_placeholder.empty()


def process_query_with_real_logging(user_input, client):
    """실제 진행사항을 기록하면서 쿼리 처리"""

    log_container = st.container()
    logger = RealTimeProcessLogger(log_container)

    try:
        logger.log_actual("INFO", "Query processing started", f"Input length: {len(user_input)}")
        
        start_time = time.time()
        hs_manager = get_hs_manager()
        load_time = time.time() - start_time
        logger.log_actual("SUCCESS", "HSDataManager loaded", f"{load_time:.2f}s")
        
        category = st.session_state.selected_category
        logger.log_actual("INFO", "Category selected", category)

        q_type = CATEGORY_MAPPING.get(category, "hs_classification")
        logger.log_actual("INFO", "Question type mapped", q_type)

        answer_start = time.time()
        
        if q_type == "web_search":
            logger.log_actual("SEARCH", "Initiating Google Search API call...")
            ai_start = time.time()
            answer = "\n\n +++ 웹검색 실시 +++\n\n" + handle_web_search(user_input, st.session_state.context, hs_manager, client)
            ai_time = time.time() - ai_start
            logger.log_actual("SUCCESS", "Web search completed", f"{ai_time:.2f}s, {len(answer)} chars")

        elif q_type == "domestic_hs_recommendation":
            # Multi-Agent 분석 실행 (UI 컨테이너 없이)
            final_answer = handle_hs_classification_cases(user_input, st.session_state.context, hs_manager, client, None)
            answer = "\n\n +++ 국내 분류사례 기반 HS 추천 +++\n\n" + final_answer

        elif q_type == "domestic_case_lookup":
            logger.log_actual("SEARCH", "Domestic case lookup starting...")
            lookup_start = time.time()
            answer = "\n\n +++ 국내 분류사례 원문 검색 +++\n\n" + handle_domestic_case_lookup(user_input, hs_manager)
            lookup_time = time.time() - lookup_start
            logger.log_actual("SUCCESS", "Domestic case lookup completed", f"{lookup_time:.2f}s, {len(answer)} chars")

        elif q_type == "overseas_hs_recommendation":
            # Multi-Agent 분석 실행 (UI 컨테이너 없이)
            final_answer = handle_overseas_hs(user_input, st.session_state.context, hs_manager, client, None)
            answer = "\n\n +++ 해외 분류사례 기반 HS 추천 +++\n\n" + final_answer

        elif q_type == "overseas_case_lookup":
            logger.log_actual("SEARCH", "Overseas case lookup starting...")
            lookup_start = time.time()
            answer = "\n\n +++ 해외 분류사례 원문 검색 +++\n\n" + handle_overseas_case_lookup(user_input, hs_manager)
            lookup_time = time.time() - lookup_start
            logger.log_actual("SUCCESS", "Overseas case lookup completed", f"{lookup_time:.2f}s, {len(answer)} chars")
            
        elif q_type == "hs_manual":
            # 1단계: HS코드 추출 시도
            logger.log_actual("INFO", "Checking for user-provided HS codes...")
            extracted_codes = extract_hs_codes(user_input)
            logger.log_actual("INFO", f"Extracted codes: {extracted_codes if extracted_codes else 'None'}")

            if extracted_codes:
                # 사용자가 HS코드를 제시한 경우 → 코드 비교 분석
                logger.log_actual("SUCCESS", f"Found {len(extracted_codes)} user-provided HS codes", ", ".join(extracted_codes))
                logger.log_actual("AI", "Starting user-provided codes comparison analysis...")
                ai_start = time.time()
                answer = "\n\n +++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++ \n\n" + handle_hs_manual_with_user_codes(user_input, st.session_state.context, hs_manager, logger, extracted_codes, client)
                ai_time = time.time() - ai_start
                logger.log_actual("SUCCESS", "User-provided codes analysis completed", f"{ai_time:.2f}s, {len(answer)} chars")
            else:
                # HS코드 없는 경우 → 에러 메시지
                logger.log_actual("ERROR", "No HS codes found in user input")
                answer = "해설서 분석 모드에서는 반드시 HS 코드를 제시해야 합니다.\n\n예시: '3923.30과 3926.90 중 어느 것이 맞나요?'"
            
        elif q_type == "hs_manual_raw":
            logger.log_actual("SEARCH", "Extracting HS codes...")
            hs_codes = extract_hs_codes(user_input)
            if hs_codes:
                logger.log_actual("SUCCESS", f"Found {len(hs_codes)} HS codes", ", ".join(hs_codes))
                logger.log_actual("DATA", "Retrieving raw HS explanations...")
                raw_start = time.time()
                raw_answer = clean_text(get_hs_explanations(hs_codes))
                raw_time = time.time() - raw_start
                answer = "\n\n +++ HS 해설서 원문 검색 실시 +++ \n\n" + raw_answer
                logger.log_actual("SUCCESS", "Raw HS manual retrieved", f"{raw_time:.2f}s, {len(raw_answer)} chars")
            else:
                logger.log_actual("ERROR", "No valid HS codes found in input")
                answer = "HS 코드를 찾을 수 없습니다. 4자리 HS 코드를 입력해주세요."

        answer_time = time.time() - answer_start
        logger.log_actual("SUCCESS", "Answer generation completed", f"{answer_time:.2f}s, {len(answer)} chars")
        
        total_time = time.time() - logger.start_time
        logger.log_actual("INFO", "Process completed successfully", f"Total time: {total_time:.2f}s")
        
        # Return the answer for external processing
        return answer
        
    except Exception as e:
        logger.log_actual("ERROR", f"Exception occurred: {str(e)}")
        logger.log_actual("ERROR", f"Error type: {type(e).__name__}")
        raise e


# 사이드바 설정 - 챗봇 특성 소개
with st.sidebar:
    st.title("🔑 API Key 설정")

    # Google API Key 입력
    user_api_key = st.text_input(
        "Google API Key를 입력하세요",
        type="password",
        help="Google AI Studio에서 발급받은 API Key를 입력하세요",
        key="google_api_key_input"
    )

    if user_api_key:
        try:
            # 사용자 입력으로 client 생성
            client = genai.Client(api_key=user_api_key)
            st.success("✅ API Key 인증 성공")
        except Exception as e:
            st.error(f"❌ API Key 오류: {str(e)}")
            st.info("올바른 API Key를 입력해주세요")
            st.stop()
    else:
        st.warning("⚠️ API Key를 입력하세요")
        st.markdown("""
        **Google API Key 발급 방법:**
        1. [Google AI Studio](https://aistudio.google.com/apikey) 접속
        2. "Create API Key" 클릭
        3. 발급된 키를 복사하여 위에 입력

        **참고:** 무료 tier에서도 충분히 사용 가능합니다.
        """)
        st.stop()

    st.divider()

    st.title("📊 HS 품목분류 전문 AI")

    st.markdown("---")

    st.markdown("""
    ### 📚 보유 데이터

    **국내 분류사례**
    - 관세청 분류사례: 899건 (10개 파일로 분할)
    - HS 위원회 결정: 76건
    - HS 협의회 결정: 12건
    - **총 987건**

    **해외 분류사례**
    - 미국 CBP 분류사례: 900건
    - EU 관세청 분류사례: 1,000건
    - **총 1,900건**

    **공식 해설서**
    - HS 품목분류표: 17,966개 코드
    - HS 해설서: 1,448개 항목
    - HS 통칙: 9개 조항

    **실시간 웹 데이터**
    - Google Search API 연동

    ---

    ### 🚀 핵심 기술

    **TF-IDF Character n-gram**
    - 복합어 정확 검색 (예: "폴리우레탄폼" 띄어쓰기 없이 인식)
    - 문서별 단어 중요도 계산
      - TF: 문서 내 반복 빈도 (예: "리튬"이 10번 등장 → 중요)
      - IDF: 희소성 가중치 (예: "제품"은 모든 문서에 등장 → 덜 중요)

    **Multi-Agent 시스템**
    - 전체 데이터(~900건)에서 TF-IDF로 상위 100개 사례 추출
    - 5그룹(각 20개 사례)으로 분할 후 병렬 분석
    - Head Agent가 5개 분석 결과 종합

    **Gemini AI 2.5 Flash**
    - Google 최신 LLM 모델
    - 빠른 응답 + 높은 정확도

    ---

    ### ⚡ 성능 특징

    - 캐싱으로 데이터 로딩 최적화
    - 실시간 처리 과정 투명 공개
    - AI 분석 결과 세션 저장
    - 대화 컨텍스트 누적 관리
    """)

    st.divider()
    
    # 새로운 채팅 시작 버튼
    if st.button("새로운 채팅 시작하기", type="primary"):
        st.session_state.chat_history = []  # 채팅 기록 초기화
        # Multi-Agent 및 HS 해설서 분석 결과도 초기화
        if 'ai_analysis_results' in st.session_state:
            st.session_state.ai_analysis_results = []
        if 'hs_manual_analysis_results' in st.session_state:
            st.session_state.hs_manual_analysis_results = []
        # 컨텍스트 초기화 (기본 컨텍스트 재사용)
        st.session_state.context = SYSTEM_PROMPT
        st.success("✅ 새로운 채팅이 시작되었습니다!")

# 메인 페이지 설정
st.title("HS 품목분류 챗봇 (API Key 입력 버전)")

# 활용 시나리오를 접을 수 있는 expander로 변경
with st.expander("💡 슬기로운 품목분류 생활 (활용 시나리오)", expanded=True):
    st.markdown("""
    <ol style='padding-left: 18px;'>
      <li style='margin-bottom: 10px;'>
        <b>[웹 검색] </span> "스마트워치 실리콘 밴드의 정확한 재질 성분과 제조 공정, 주요 용도는 무엇인가?"</b><br>
        <span style='color:#059669;'>✓ 답변: 합성고무(실리콘), 스마트워치 전용 부속품 확인</span>
      </li>
      <li style='margin-bottom: 10px;'>
        <b>[국내 사례] </span> "실리콘 재질로 만든 스마트워치용 교체 밴드는 어떤 HS코드로 분류되나요?"</b><br>
        <span style='color:#059669;'>✓ 답변: 9113.90 (시계 부속품) 12건 / 3926.90 (플라스틱) 5건</span>
      </li>
      <li style='margin-bottom: 10px;'>
        <b>[해외 사례] </span> "미국과 EU에서 스마트워치 실리콘 스트랩(watch strap)의 분류 사례와 관세율을 알려줘"</b><br>
        <span style='color:#059669;'>✓ 답변: 전 세계 동일: 9113.90 분류</span>
      </li>
      <li>
        <b>[해설서 분석] </span> "스마트워치 전용 실리콘 밴드가 9113.90 시계 부속품과 3926.90 기타 플라스틱 제품 중 어디에 분류되는지 해설서와 통칙을 근거로 비교 분석해줘"</b><br>
        <span style='color:#059669;'>✓ 결론: 9113.90-0000 (통칙 1, 구체성 원칙)</span>
      </li>
    </ol>
    """, unsafe_allow_html=True)

# 질문 유형 선택 라디오 버튼 (가로 배치)
st.subheader("질문 유형을 선택하세요")

selected_category = st.radio(
    "분석 방식을 선택하세요:",
    [
        "웹 검색",
        "국내 분류사례 기반 HS 추천",
        "해외 분류사례 기반 HS 추천",
        "HS해설서 분석(품명 + 후보 HS코드)",
        "국내 분류사례 원문 검색",
        "해외 분류사례 원문 검색",
        "HS해설서 원문 검색(HS코드만 입력)"
    ],
    index=0,
    horizontal=True,
    key="category_radio",
    label_visibility="collapsed"
)

# 카테고리명은 그대로 사용
st.session_state.selected_category = selected_category

# 특수 케이스 주의사항 표시
if selected_category == "HS해설서 분석(품명 + 후보 HS코드)":
    st.warning("주의: 반드시 비교할 HS코드를 질문에 포함해야 합니다.")

st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)  # 간격 축소

# 채팅 기록 표시
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""<div style='background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                   <strong>사용자:</strong> {message['content']}
                   </div>""", unsafe_allow_html=True)
    else:
        # 분석 과정이 있는 경우 expander 표시
        if any(keyword in message['content'] for keyword in ["+++ 국내 분류사례 기반 HS 추천 +++", "+++ 해외 분류사례 기반 HS 추천 +++", "+++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++"]):
            # AI 분석 과정 expander 표시 (닫힌 상태)
            with st.expander("🔍 **AI 분석 과정 보기**", expanded=False):
                if "+++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++" in message['content']:
                    # 사용자 제시 코드 분석의 경우
                    st.info("🔍 **사용자 제시 HS코드 기반으로 분석되었습니다**")
                    st.markdown("""
                    **분석 과정:**
                    1. 📝 사용자 질문에서 HS코드 추출
                    2. 📊 각 HS코드별 품목분류표 정보 수집
                    3. 📚 각 HS코드별 해설서 정보 수집 및 요약
                    4. 📋 HS 분류 통칙 준비
                    5. 🧠 최종 AI 비교 분석 (Gemini 2.5)
                    """)
                elif st.session_state.ai_analysis_results:
                    # Multi-Agent 분석의 경우 - 쿼리 확장 결과 표시 (제일 위)
                    if hasattr(st.session_state, 'query_expansion_result') and st.session_state.query_expansion_result:
                        exp_result = st.session_state.query_expansion_result
                        st.success("✅ **AI 쿼리 확장 완료**")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**원본 쿼리:** {exp_result['original_query']}")
                            st.markdown(f"**식별된 물품:** {exp_result['target_product']}")
                            if exp_result.get('material'):
                                st.markdown(f"**재질:** {exp_result['material']}")
                            if exp_result.get('components'):
                                st.markdown(f"**주요 성분:** {exp_result['components']}")
                            if exp_result.get('function'):
                                st.markdown(f"**주요 기능:** {exp_result['function']}")

                        with col2:
                            keyword_groups = exp_result.get('keyword_groups', {})
                            if keyword_groups.get('similar_korean'):
                                st.markdown(f"**한글 유사어:** {', '.join(keyword_groups['similar_korean'][:5])}")
                            if keyword_groups.get('similar_english'):
                                st.markdown(f"**영문 유사어:** {', '.join(keyword_groups['similar_english'][:5])}")
                            if keyword_groups.get('material'):
                                st.markdown(f"**재질 관련 용어:** {', '.join(keyword_groups['material'][:3])}")
                            if keyword_groups.get('component'):
                                st.markdown(f"**성분 관련 용어:** {', '.join(keyword_groups['component'][:3])}")
                            if keyword_groups.get('function'):
                                st.markdown(f"**기능 관련 용어:** {', '.join(keyword_groups['function'][:3])}")

                        with st.expander("🔎 **전체 확장된 쿼리 보기**", expanded=False):
                            st.text(exp_result['expanded_query'])

                        st.divider()

                    # 저장된 그룹별 분석 결과 표시
                    for result in st.session_state.ai_analysis_results:
                        emoji = "🤖" if result['type'] == 'domestic' else "🌐"
                        st.success(f"{emoji} **그룹 {result['group_id']+1} AI 분석 완료** ({result['processing_time']:.1f}초)")
                        with st.container():
                            st.write(f"⏰ {result['start_time']}")
                            st.markdown("**분석 결과:**")
                            st.info(result['answer'])
                            st.divider()
                else:
                    st.info("분석 과정 정보가 저장되지 않았습니다.")

            # 최종 답변은 expander 외부에 항상 표시 (마크다운으로 렌더링)
            st.markdown("**품목분류 전문가:**")
            st.markdown(message['content'])

        # HS 해설서 원문인지 확인
        elif "+++ HS 해설서 원문 검색 실시 +++" in message['content']:
            # 마크다운으로 렌더링하여 구조화된 형태로 표시
            st.markdown("**품목분류 전문가:**")
            st.markdown(message['content'])
        else:
            st.markdown(f"""<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                    <strong>품목분류 전문가:</strong> {message['content']}
                    </div>""", unsafe_allow_html=True)


# 하단 입력 영역 (Form 기반 입력)
input_container = st.container()
st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)

with input_container:
    # 선택된 카테고리의 예시 질문들 (텍스트로만 표시)
    examples = EXAMPLE_QUESTIONS.get(st.session_state.selected_category, [])

    if examples:
        st.markdown("**💡 예시 질문:**")
        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx]:
                st.markdown(f"📌 {example}")

    # Form을 사용하여 안정적인 입력 처리
    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_input(
            "품목에 대해 질문하세요:",
            placeholder="여기에 입력 후 Enter 또는 전송 버튼 클릭"
        )
        
        # 두 개의 컬럼으로 나누어 버튼을 오른쪽에 배치
        col1, col2 = st.columns([4, 1])
        with col2:
            submit_button = st.form_submit_button("전송", use_container_width=True)
        
        # 폼이 제출되고 입력값이 있을 때 처리
        if submit_button and user_input and user_input.strip():
            selected_category = st.session_state.selected_category

            # HS Manager 인스턴스 가져오기
            hs_manager = get_hs_manager()
            
            # 분석 과정 표시가 필요한 유형들
            if selected_category in ["국내 분류사례 기반 HS 추천", "해외 분류사례 기반 HS 추천", "HS해설서 분석(품명 + 후보 HS코드)"]:
                if selected_category in ["국내 분류사례 기반 HS 추천", "해외 분류사례 기반 HS 추천"]:
                    st.session_state.ai_analysis_results = []  # Multi-Agent용 결과 초기화
                analysis_expander = st.expander("🔍 **AI 분석 과정 보기**", expanded=True)

            try:
                # 분석 과정 표시 방식 분기
                if selected_category == "HS해설서 분석(품명 + 후보 HS코드)":
                    # HS 해설서 분석은 HS코드 유무에 따라 분기
                    class DummyLogger:
                        def log_actual(self, level, message, data=None):
                            pass  # UI 표시용이므로 로깅은 생략

                    dummy_logger = DummyLogger()
                    extracted_codes = extract_hs_codes(user_input)

                    if extracted_codes:
                        # HS코드가 있으면 사용자 제시 코드 비교 분석
                        final_answer = handle_hs_manual_with_user_codes(user_input, st.session_state.context, hs_manager, dummy_logger, extracted_codes, client, analysis_expander)
                        answer = "\n\n +++ HS 해설서 분석 실시 (사용자 제시 코드 비교) +++ \n\n" + final_answer
                    else:
                        # HS코드가 없으면 에러 메시지
                        answer = "해설서 분석 모드에서는 반드시 HS 코드를 제시해야 합니다.\n\n예시: '3923.30과 3926.90 중 어느 것이 맞나요?'"
                elif selected_category not in ["국내 분류사례 기반 HS 추천", "해외 분류사례 기반 HS 추천"]:
                    # 기타 유형은 로그 패널 표시
                    with st.expander("실시간 처리 과정 로그 보기", expanded=True):
                        answer = process_query_with_real_logging(user_input, client)
                else:
                    # Multi-Agent 분석용 특별 처리
                    if selected_category == "국내 분류사례 기반 HS 추천":
                        # utils 함수를 직접 호출하되 expander 컨테이너 전달
                        final_answer = handle_hs_classification_cases(user_input, st.session_state.context, hs_manager, client, analysis_expander)
                        answer = "\n\n +++ 국내 분류사례 기반 HS 추천 +++\n\n" + final_answer
                    elif selected_category == "해외 분류사례 기반 HS 추천":
                        final_answer = handle_overseas_hs(user_input, st.session_state.context, hs_manager, client, analysis_expander)
                        answer = "\n\n +++ 해외 분류사례 기반 HS 추천 +++\n\n" + final_answer

                # Update chat history after successful processing
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.context += f"\n사용자: {user_input}\n품목분류 전문가: {answer}\n"

                # 분석 과정이 표시된 유형들의 최종 답변 표시 (마크다운으로 렌더링)
                if selected_category in ["국내 분류사례 기반 HS 추천", "해외 분류사례 기반 HS 추천", "HS해설서 분석(품명 + 후보 HS코드)"]:
                    st.markdown("**품목분류 전문가:**")
                    st.markdown(answer)
                
                # Force rerun to display the new chat messages
                st.rerun()
                
            except APIError as e:
                st.error("### Gemini API 오류 발생")
                st.error(f"**오류 코드**: {e.code}")
                st.error(f"**오류 메시지**: {e.message}")

                if e.code == 503:
                    st.warning("**해결 방법**: API 서버가 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요.")
                elif e.code == 429:
                    st.warning("**해결 방법**: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
                elif e.code == 404:
                    st.warning("**해결 방법**: 요청한 모델을 찾을 수 없습니다. 모델명을 확인해주세요.")
                elif e.code == 400:
                    st.warning("**해결 방법**: 잘못된 요청입니다. 입력 내용을 확인해주세요.")
                else:
                    st.warning("**해결 방법**: 문제가 지속되면 관리자에게 문의해주세요.")

            except Exception as e:
                st.error(f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}")