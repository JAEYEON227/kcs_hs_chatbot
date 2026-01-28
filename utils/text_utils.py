import json
import re
from typing import List

# HTML 태그 제거 및 텍스트 정제 함수
def clean_text(text):
    """HTML 태그 제거 및 텍스트 정제"""
    # HTML 태그 제거 (더 엄격한 정규식 패턴 사용)
    text = re.sub(r'<[^>]+>', '', text)  # 모든 HTML 태그 제거
    text = re.sub(r'\s*</div>\s*$', '', text)  # 끝에 있는 </div> 태그 제거
    return text.strip()

# HS 코드 추출 패턴 정의 및 함수
# 더 유연한 HS 코드 추출 패턴
HS_PATTERN = re.compile(
    r'(?:HS\s*)?(\d{4}(?:[.-]?\d{2}(?:[.-]?\d{2}(?:[.-]?\d{2})?)?)?)',
    flags=re.IGNORECASE
)

def extract_hs_codes(text):
    """
    여러 HS 코드를 추출하고, 중복 제거 및 숫자만 남겨 표준화
    개선사항:
    - 단어 경계(\b) 제거로 더 유연한 매칭
    - 숫자만 있는 경우도 처리 가능
    - 최소 4자리 숫자 체크 추가
    """
    matches = HS_PATTERN.findall(text)
    hs_codes = []

    for raw in matches:
        # 숫자만 남기기
        code = re.sub(r'\D', '', raw)
        # 최소 4자리이고 중복이 아닌 경우만 추가
        if len(code) >= 4 and code not in hs_codes:
            hs_codes.append(code)

    # 만약 위 패턴으로 찾지 못하고, 입력이 4자리 이상의 숫자로만 구성된 경우
    if not hs_codes:
        # 순수 숫자만 있는 경우 체크
        numbers_only = re.findall(r'\d{4,}', text)
        for num in numbers_only:
            if num not in hs_codes:
                hs_codes.append(num)

    return hs_codes

def extract_and_store_text(json_file):
    """JSON 파일에서 head1과 text를 추출하여 변수에 저장"""
    try:
        # JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 데이터를 변수에 저장
        extracted_data = []
        for item in data:
            head1 = item.get('head1', '')
            text = item.get('text', '')
            if head1 or text:
                extracted_data.append(f"{head1}\n{text}")

        return extracted_data
    except Exception as e:
        print(f"오류 발생: {e}")
        return []

# 통칙 데이터 로드 (재사용을 위한 전역 변수)
general_explanation = extract_and_store_text('knowledge/통칙_grouped.json')