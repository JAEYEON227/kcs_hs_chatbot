import json
import re
def lookup_hscode(hs4: str, json_path: str):
    """
    hs4: HS 4자리 문자열, 예) '8517'
    json_path: grouped JSON 파일 경로
    returns: (part_entry, chapter_entry, sub_entry)
    """
    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        groups = json.load(f)

    # 1) 류(類) key: "제00류"
    chapter_key = f"제{int(hs4[:2])}류"
    chapter_entry = next((g for g in groups if g.get('header2') == chapter_key), None)

    # 2) 소호 key: "00.00"
    sub_key = f"{hs4[:2]}.{hs4[2:]}"
    sub_entry = next((g for g in groups if g.get('header2') == sub_key), None)

    # 3) 부(部) key: "제00부"
    part_key = chapter_entry.get('header1') if chapter_entry else None
    part_entry = next((g for g in groups if (g.get('header1') == part_key)&(re.sub(r'제\s*(\d+)\s*부', r'제\1부', g.get('header1')) == part_key)), None)

    return part_entry, chapter_entry, sub_entry

if __name__ == '__main__':
    # 예시: 사용자 입력
    hs_code = input("HS 코드 네 자리 입력 (예: 8517): ").strip()
    json_file = 'knowledge/grouped_11_end.json'  # 실제 JSON 파일 경로로 수정

    part, chapter, sub = lookup_hscode(hs_code, json_file)

    print("\n=== 부(部) 항목 ===")
    print(json.dumps(part, ensure_ascii=False, indent=2) if part else "찾을 수 없습니다.")

    print("\n=== 류(類) 항목 ===")
    print(json.dumps(chapter, ensure_ascii=False, indent=2) if chapter else "찾을 수 없습니다.")

    print(f"\n=== 소호({hs_code[:2]}.{hs_code[2:]}) 항목 ===")
    print(json.dumps(sub, ensure_ascii=False, indent=2) if sub else "찾을 수 없습니다.")
