# 관세율표 제작
# 관세청_HS별 관세율표_20150331 (공공데이터포털)
# 관세쳥_품목번호별 관세율표_20250519 (공공데이터포털)

import pandas as pd

def step1_load_a(csv_path):
    print("▶ Step 1: A 데이터 로딩 시작")
    df = pd.read_csv(csv_path, encoding='cp949', dtype={'세번': str})
    # ── 여기를 추가: 순번 1~1619까지 '세번' 앞에 '0'을 붙여 줍니다.
    mask = df['순번'] <= 1619
    df.loc[mask, '세번'] = '0' + df.loc[mask, '세번']
    df = df[['세번', '영문품명', '한글품명']].drop_duplicates()
    df.rename(columns={'세번': '품목번호'}, inplace=True)
    print(df.head(10))
    print(f"  완료: {len(df)}행, {df.columns.tolist()} 컬럼")
    return df

def step2_load_b(excel_path):
    print("▶ Step 2: B 데이터 로딩 시작")
    df = pd.read_excel(excel_path, sheet_name='5.19', dtype={'품목번호': str}, engine='openpyxl')
    print(df.head(10))
    print(f"  완료: {len(df)}행, {df.columns.tolist()} 컬럼")
    return df

def step3_pivot_b(df_b):
    print("▶ Step 3: B 데이터 Pivot 시작")
    df_wide = df_b.pivot_table(
        index='품목번호',
        columns='관세율구분',
        values='관세율',
        aggfunc='first'
    ).reset_index()
    print(f"  완료: {df_wide.shape[0]}행 × {df_wide.shape[1]}열")
    print(df_wide.head(10))
    return df_wide

def step4_merge(df_a, df_b_wide):
    print("▶ Step 4: 데이터 병합 시작")
    merged = pd.merge(df_a, df_b_wide, on='품목번호', how='left')
    print(f"  완료: {merged.shape[0]}행")
    return merged

def step5_export(df, path):
    import csv
    print("▶ Step 5: 결과 저장 시작")
    df.to_csv(path, encoding='utf-8-sig', index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"  완료: '{path}'에 저장됨 ({df.shape[0]}행 × {df.shape[1]}열)")

def step6_export_json(df, path):
    print("▶ Step 6: JSON 파일 저장 시작")
    df_json = df[["품목번호", "영문품명", "한글품명"]]
    df_json.to_json(path, orient='records', force_ascii=False, indent=4)
    print(f"  완료: '{path}'에 저장됨 ({len(df_json)}행)")

if __name__ == "__main__":
    a = step1_load_a("./품목분류표_제작/HS별 관세율표.csv")
    b = step2_load_b("./품목분류표_제작/품목번호별 관세율(2025).xlsx")
    b_wide = step3_pivot_b(b)
    merged = step4_merge(a, b_wide)
    print(merged.head(20))
    step5_export(merged, "./knowledge/hstable.csv")
    step6_export_json(merged, "./knowledge/hstable.json")
    print("🎉 전체 프로세스 완료")

