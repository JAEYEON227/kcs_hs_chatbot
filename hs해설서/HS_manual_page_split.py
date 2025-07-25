import pdfplumber
import json

# 입력 PDF 경로
input_path = 'HS해설서_국문.pdf'

# 1. 통칙 페이지 JSON 생성
tongchik_output = '통칙_grouped.json'
tongchik_groups = {}

with pdfplumber.open(input_path) as pdf:
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        # 첫 줄이 "통칙"인 페이지만 처리
        if not lines or lines[0] != "통칙":
            continue
        header1 = lines[0]
        header2 = lines[1] if len(lines) > 1 else ""
        key = (header1, header2)
        if key not in tongchik_groups:
            tongchik_groups[key] = {
                "header1": header1,
                "header2": header2,
                "pages": [],
                "text": ""
            }
        page_num = idx + 1
        tongchik_groups[key]["pages"].append(page_num)
        tongchik_groups[key]["text"] += f"\n--- Page {page_num} ---\n{text}"

tongchik_result = list(tongchik_groups.values())
with open(tongchik_output, 'w', encoding='utf-8') as f:
    json.dump(tongchik_result, f, ensure_ascii=False, indent=2)

# 2. 11페이지부터 끝까지 그룹화하여 JSON 생성
rest_output = 'grouped_11_end.json'
rest_groups = {}

with pdfplumber.open(input_path) as pdf:
    for idx in range(10, len(pdf.pages)):
        page = pdf.pages[idx]
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        header1 = lines[0] if len(lines) >= 1 else ""
        header2 = lines[1] if len(lines) >= 2 else ""
        key = (header1, header2)
        if key not in rest_groups:
            rest_groups[key] = {
                "header1": header1,
                "header2": header2,
                "pages": [],
                "text": ""
            }
        page_num = idx + 1
        rest_groups[key]["pages"].append(page_num)
        rest_groups[key]["text"] += f"\n--- Page {page_num} ---\n{text}"

rest_result = list(rest_groups.values())
with open(rest_output, 'w', encoding='utf-8') as f:
    json.dump(rest_result, f, ensure_ascii=False, indent=2)

# 요약 출력
print(f"통칙 그룹 수: {len(tongchik_result)}, 파일 생성: {tongchik_output}")
print(f"나머지 그룹 수: {len(rest_result)}, 파일 생성: {rest_output}")
