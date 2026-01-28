# 카테고리 -> 질문 타입 매핑
CATEGORY_MAPPING = {
    "웹 검색": "web_search",
    "국내 분류사례 기반 HS 추천": "domestic_hs_recommendation",
    "국내 분류사례 원문 검색": "domestic_case_lookup",
    "해외 분류사례 기반 HS 추천": "overseas_hs_recommendation",
    "해외 분류사례 원문 검색": "overseas_case_lookup",
    "HS해설서 분석(품명 + 후보 HS코드)": "hs_manual",
    "HS해설서 원문 검색(HS코드만 입력)": "hs_manual_raw"
}

# 로거 아이콘
LOGGER_ICONS = {
    "INFO": "ℹ️",
    "SUCCESS": "✅",
    "ERROR": "❌",
    "DATA": "📊",
    "AI": "🤖",
    "SEARCH": "🔍"
}

# 예시 질문
EXAMPLE_QUESTIONS = {
    "웹 검색": [
        "전기차 배터리 최신 기술 개발 현황은?",
        "반도체 시장 동향과 주요 수출국은?",
        "수소연료전지의 글로벌 시장 전망은?"
    ],
    "국내 분류사례 기반 HS 추천": [
        "자동차에 사용되는 플라스틱 부품의 품목분류 사례는?",
        "기능성 화장품(미백, 주름개선)의 HS코드 분류 사례를 알려주세요",
        "반도체 제조장비 부분품의 품목분류 사례는?"
    ],
    "국내 분류사례 원문 검색": [
        "품목분류2과-9433",
        "footwear 관련 사례",
        "동기모터 분류사례"
    ],
    "해외 분류사례 기반 HS 추천": [
        "미국에서 footwear with rubber sole and leather upper의 분류 사례를 알려주세요",
        "미국의 toy 분류 사례와 관세율은?",
        "knit fabric의 해외 분류 사례는?"
    ],
    "해외 분류사례 원문 검색": [
        "NY N338825",
        "footwear 미국 사례",
        "toy 사례"
    ],
    "HS해설서 분석(품명 + 후보 HS코드)": [
        "6403.99 가죽신발과 6402.99 고무신발 중 고무밑창 가죽갑피 신발은?",
        "3917.32와 4009 중 폴리아미드 호스는?",
        "8501.10과 8501.40 중 동기모터는?"
    ],
    "HS해설서 원문 검색(HS코드만 입력)": [
        "6403",
        "3917",
        "8501"
    ]
}
