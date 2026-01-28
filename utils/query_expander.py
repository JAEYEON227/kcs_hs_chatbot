"""
AI 기반 쿼리 확장 모듈
- 사용자 쿼리에서 물품 식별
- 품목분류 용어사전 기반 유사어 생성
- 확장된 키워드 그룹 반환
"""

import json
import os
from typing import List, Dict, Set
from google import genai
from google.genai.errors import APIError



class QueryExpander:
    """
    AI 기반 쿼리 확장기

    Balanced 용어사전(1,149개 단어)을 활용하여
    사용자 쿼리를 확장된 키워드 그룹으로 변환
    """

    def __init__(self, client, terminology_version='balanced'):
        """
        Args:
            client: Google Gemini API client
            terminology_version: 'minimal', 'balanced', 'full' 중 선택
        """
        self.client = client
        self.terminology = self._load_terminology(terminology_version)
        self.terminology_version = terminology_version

    def _load_terminology(self, version):
        """용어사전 로드"""
        terminology_file = f'knowledge/hs_terminology_{version}.json'

        if not os.path.exists(terminology_file):
            print(f"Warning: {terminology_file} not found. Query expansion disabled.")
            return None

        with open(terminology_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Query Expander initialized with '{version}' terminology")
        print(f"  - Total terms: {data['metadata']['total_terms']}")
        print(f"  - Coverage: {data['metadata']['coverage_rate']}%")

        return data

    def _create_expansion_prompt(self, user_query: str) -> str:
        """쿼리 확장을 위한 프롬프트 생성"""

        if not self.terminology:
            return None

        # 용어사전에서 샘플 추출 (Few-shot 학습용)
        sample_terms = self.terminology['terms'][:100]  # 처음 100개만

        prompt = f"""당신은 HS 품목분류 전문가입니다.

**임무**: 사용자 질문을 체계적으로 분석하여 HS 코드 검색에 필요한 모든 키워드를 생성하세요.

**분석 단계**:
1. **물품 식별**: 사용자가 HS 코드를 알고 싶어하는 물품명
2. **재질 식별**: 물품의 재질, 소재 (예: 플라스틱, 금속, 섬유 등)
3. **성분 식별**: 물품의 주요 성분, 화학물질 (예: 폴리에틸렌, 리튬, 면 등)
4. **기능 식별**: 물품의 주요 기능, 용도 (예: 보호, 장식, 전원공급 등)
5. **유사어 생성**: 위 내용을 바탕으로 한글+영문 유사어 생성

**제약 조건**:
1. 반드시 아래 "품목분류 용어사전"에 있는 단어들만 사용하세요
2. 용어사전에 없는 단어는 절대 생성하지 마세요
3. 원본 쿼리의 핵심 키워드도 함께 포함하세요
4. **영문 유사어를 반드시 포함하세요** (용어사전에서 찾아서 사용)

**품목분류 용어사전 (샘플 - 총 {self.terminology['metadata']['total_terms']}개)**:
{', '.join(sample_terms)}
... (이하 생략)

**출력 형식** (JSON):
{{
  "target_product": "식별된 물품명",
  "material": "재질 (예: 실리콘, 플라스틱, 금속 등)",
  "components": "주요 성분 (예: 폴리에틸렌, 리튬, 면 등)",
  "function": "주요 기능 (예: 보호, 장식, 전원공급 등)",
  "original_keywords": ["원본", "키워드"],
  "similar_terms_korean": ["한글유사어1", "한글유사어2", ...],
  "similar_terms_english": ["english1", "english2", ...],
  "material_terms": ["재질1", "재질2", "material1", "material2", ...],
  "component_terms": ["성분1", "성분2", "component1", "component2", ...],
  "function_terms": ["기능1", "기능2", "function1", "function2", ...],
  "expanded_query": "확장된 검색 쿼리 (모든 키워드 한글+영문 결합)"
}}

**예시 1**:
사용자: "리튬이온 배터리로 작동하는 스마트워치용 실리콘 밴드"

출력:
{{
  "target_product": "스마트워치 밴드",
  "material": "실리콘 (합성고무)",
  "components": "실리콘 폴리머",
  "function": "시계 착용, 손목 고정",
  "original_keywords": ["리튬이온", "배터리", "스마트워치", "실리콘", "밴드"],
  "similar_terms_korean": ["시계", "줄", "스트랩", "손목밴드"],
  "similar_terms_english": ["watch", "band", "strap", "wrist", "wearable"],
  "material_terms": ["실리콘", "고무", "플라스틱", "합성수지", "silicone", "rubber", "plastic"],
  "component_terms": ["폴리머", "polymer", "synthetic"],
  "function_terms": ["시계", "부속품", "착용", "watch", "accessory", "parts"],
  "expanded_query": "스마트워치 밴드 시계 줄 스트랩 손목밴드 실리콘 고무 플라스틱 합성수지 폴리머 부속품 watch band strap wrist silicone rubber plastic polymer accessory parts"
}}

**예시 2**:
사용자: "폴리에틸렌 재질의 일회용 비닐봉지"

출력:
{{
  "target_product": "비닐봉지",
  "material": "폴리에틸렌 (플라스틱)",
  "components": "폴리에틸렌",
  "function": "포장, 운반",
  "original_keywords": ["폴리에틸렌", "일회용", "비닐봉지"],
  "similar_terms_korean": ["봉지", "포장", "비닐"],
  "similar_terms_english": ["bag", "plastic", "packaging", "poly"],
  "material_terms": ["폴리에틸렌", "플라스틱", "합성수지", "polyethylene", "plastic"],
  "component_terms": ["에틸렌", "폴리머", "ethylene", "polymer"],
  "function_terms": ["포장", "운반", "저장", "packaging", "carrying", "bag"],
  "expanded_query": "비닐봉지 봉지 포장 비닐 폴리에틸렌 플라스틱 합성수지 에틸렌 폴리머 운반 저장 bag plastic packaging poly polyethylene ethylene polymer carrying"
}}

이제 아래 사용자 질문을 분석하세요:

사용자: {user_query}

출력 (JSON만):"""

        return prompt

    def expand_query(self, user_query: str) -> Dict:
        """
        사용자 쿼리를 확장하여 키워드 그룹 생성

        Args:
            user_query: 사용자의 원본 질문

        Returns:
            {
                'original_query': 원본 쿼리,
                'target_product': 식별된 물품,
                'expanded_query': 확장된 쿼리,
                'keyword_groups': {
                    'original': [...],
                    'similar': [...],
                    'material': [...],
                    'usage': [...]
                },
                'all_keywords': [...],  # 중복 제거된 전체 키워드
                'expansion_applied': True/False
            }
        """

        # 용어사전이 없으면 원본 쿼리만 반환
        if not self.terminology:
            return {
                'original_query': user_query,
                'target_product': None,
                'expanded_query': user_query,
                'keyword_groups': {'original': [user_query]},
                'all_keywords': [user_query],
                'expansion_applied': False
            }

        # Fallback 모델 리스트 (우선순위 순)
        # 1. Primary: Gemini 2.5 Flash Lite (우선 사용 - 고속/경량)
        # 2. Fallback: Gemini 2.0 Flash (백업)
        fallback_models = ["gemini-2.5-flash-lite", "gemini-2.0-flash"]
        
        last_error = None
        
        for model_name in fallback_models:
            try:
                # AI 기반 쿼리 확장
                prompt = self._create_expansion_prompt(user_query)

                # 단일 모델 재시도 없이 1회 시도 (Fallback으로 빠른 전환 유도)
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                # JSON 파싱
                response_text = response.text.strip()

                # JSON 블록 추출 (```json ... ``` 형식 처리)
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                expansion_data = json.loads(response_text)

                # 모든 키워드 수집 (중복 제거) - 새로운 JSON 구조 반영
                all_keywords = set()
                all_keywords.update(expansion_data.get('original_keywords', []))
                all_keywords.update(expansion_data.get('similar_terms_korean', []))
                all_keywords.update(expansion_data.get('similar_terms_english', []))
                all_keywords.update(expansion_data.get('material_terms', []))
                all_keywords.update(expansion_data.get('component_terms', []))
                all_keywords.update(expansion_data.get('function_terms', []))

                # 결과 구성 (성공 시 반환)
                return {
                    'original_query': user_query,
                    'target_product': expansion_data.get('target_product', ''),
                    'material': expansion_data.get('material', ''),
                    'components': expansion_data.get('components', ''),
                    'function': expansion_data.get('function', ''),
                    'expanded_query': expansion_data.get('expanded_query', user_query),
                    'keyword_groups': {
                        'original': expansion_data.get('original_keywords', []),
                        'similar_korean': expansion_data.get('similar_terms_korean', []),
                        'similar_english': expansion_data.get('similar_terms_english', []),
                        'material': expansion_data.get('material_terms', []),
                        'component': expansion_data.get('component_terms', []),
                        'function': expansion_data.get('function_terms', [])
                    },
                    'all_keywords': list(all_keywords),
                    'expansion_applied': True
                }

            except Exception as e:
                # 현재 모델 실패, 에러 기록하고 다음 모델 시도
                last_error = e
                print(f"Query expansion failed with {model_name}: {str(e)}")
                continue

        # 모든 모델 실패 시 최종 Fallback (원본 쿼리 반환)
        print("All query expansion models failed. Falling back to original query...")
        return {
            'original_query': user_query,
            'target_product': None,
            'expanded_query': user_query,
            'keyword_groups': {'original': [user_query]},
            'all_keywords': [user_query],
            'expansion_applied': False,
            'error': str(last_error) if last_error else "Unknown Error"
        }

    def expand_query_simple(self, user_query: str) -> str:
        """
        간단한 인터페이스: 확장된 쿼리 문자열만 반환

        Args:
            user_query: 원본 쿼리

        Returns:
            확장된 쿼리 문자열
        """
        result = self.expand_query(user_query)
        return result['expanded_query']

    def get_all_keywords(self, user_query: str) -> List[str]:
        """
        모든 키워드 리스트 반환 (중복 제거)

        Args:
            user_query: 원본 쿼리

        Returns:
            키워드 리스트
        """
        result = self.expand_query(user_query)
        return result['all_keywords']


def create_query_expander(client, version='balanced'):
    """
    QueryExpander 인스턴스 생성 헬퍼 함수

    Args:
        client: Gemini API client
        version: 'minimal', 'balanced', 'full'

    Returns:
        QueryExpander 인스턴스
    """
    return QueryExpander(client, version)


# 테스트 코드
if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    from google import genai

    # 환경 변수 로드
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # QueryExpander 생성
    expander = QueryExpander(client, 'balanced')

    # 테스트 쿼리
    test_queries = [
        "리튬이온 배터리로 작동하는 스마트워치용 실리콘 밴드",
        "폴리에틸렌 재질의 비닐봉지",
        "강화유리로 만든 스마트폰 액정보호필름"
    ]

    print("\n" + "="*60)
    print("쿼리 확장 테스트")
    print("="*60)

    for query in test_queries:
        print(f"\n[원본 쿼리]: {query}")
        result = expander.expand_query(query)

        if result['expansion_applied']:
            print(f"[식별된 물품]: {result['target_product']}")
            print(f"[재질]: {result.get('material', 'N/A')}")
            print(f"[주요 성분]: {result.get('components', 'N/A')}")
            print(f"[주요 기능]: {result.get('function', 'N/A')}")
            print(f"[한글 유사어]: {', '.join(result['keyword_groups'].get('similar_korean', [])[:5])}")
            print(f"[영문 유사어]: {', '.join(result['keyword_groups'].get('similar_english', [])[:5])}")
            print(f"[확장된 쿼리]: {result['expanded_query'][:150]}...")
            print(f"[전체 키워드 수]: {len(result['all_keywords'])}개")
        else:
            print(f"[확장 실패]: {result.get('error', '알 수 없는 오류')}")
        print(f"[확장 적용]: {result['expansion_applied']}")
