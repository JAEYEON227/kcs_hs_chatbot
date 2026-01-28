# 질문 유형 분류 함수 (LLM 기반)
def classify_question(user_input, client):
    """
    LLM(Gemini)을 활용하여 사용자의 질문을 아래 네 가지 유형 중 하나로 분류합니다.
    - 'web_search': 물품 개요, 용도, 기술개발, 무역동향, 산업동향 등
    - 'hs_classification': HS 코드, 품목분류, 관세 등
    - 'hs_manual': HS 해설서 본문 심층 분석
    - 'overseas_hs': 해외(미국/EU) HS 분류 사례
    """
    system_prompt = """
아래는 HS 품목분류 전문가를 위한 질문 유형 분류 기준입니다.

질문 유형:
1. "web_search" : "뉴스", "최근", "동향", "해외", "산업, 기술, 무역동향" 등 일반 정보 탐색이 필요한 경우.
2. "hs_classification": HS 코드, 품목분류, 관세, 세율 등 HS 코드 관련 정보가 필요한 경우.
3. "hs_manual": HS 해설서 본문 심층 분석이 필요한 경우.
4. "overseas_hs": "미국", "해외", "외국", "US", "America", "EU", "유럽" 등 해외 HS 분류 사례가 필요한 경우.
5. "hs_manual_raw": HS 코드만 입력하여 해설서 원문을 보고 싶은 경우.

아래 사용자 질문을 읽고, 반드시 위 다섯 가지 중 하나의 유형만 한글이 아닌 소문자 영문으로 답변하세요.
질문: """ + user_input + """\n답변:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash", # 또는 최신 모델로 변경 가능
        contents=system_prompt,
        )
    answer = response.text.strip().lower()
    # 결과가 정확히 네 가지 중 하나인지 확인
    if answer in ["web_search", "hs_classification", "hs_manual", "overseas_hs", "hs_manual_raw"]:
        return answer
    # 예외 처리: 분류 실패 시 기본값
    return "hs_classification"