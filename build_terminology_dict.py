"""
HS 품목분류 용어사전 구축 스크립트
전략: B(빈도) + C(TF-IDF) + D(계층적 샘플링)
"""

import json
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


def load_hstable():
    """관세율표 데이터 로드"""
    with open('knowledge/hstable.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def tokenize_korean(text):
    """한글 텍스트 토큰화"""
    if not text:
        return []
    tokens = re.findall(r'[가-힣]+', text)
    return [t for t in tokens if len(t) >= 2]


def tokenize_english(text):
    """영문 텍스트 토큰화"""
    if not text:
        return []
    text = text.lower()
    tokens = re.findall(r'[a-z]+', text)
    return [t for t in tokens if len(t) >= 3]


def get_stopwords():
    """불용어 리스트"""
    korean_stopwords = {'기타', '제품', '물품', '상품', '부분품', '부속품', '따로', '분류되지', '않은'}
    english_stopwords = {'the', 'and', 'or', 'of', 'for', 'with', 'not', 'other',
                         'parts', 'accessories', 'articles', 'goods', 'products',
                         'elsewhere', 'specified', 'included'}
    return korean_stopwords, english_stopwords


def extract_all_terms(hstable_data):
    """1단계: 전체 단어 추출 및 빈도 계산"""
    print("\n" + "="*60)
    print("1단계: 전체 단어 추출 및 빈도 분석")
    print("="*60)

    korean_stopwords, english_stopwords = get_stopwords()

    term_frequency = defaultdict(int)
    term_to_codes = defaultdict(set)
    korean_terms = set()
    english_terms = set()

    for item in hstable_data:
        hs_code = item.get('품목번호', '')
        korean_name = item.get('한글품명', '')
        english_name = item.get('영문품명', '')

        # 한글 처리
        korean_tokens = tokenize_korean(korean_name)
        for token in korean_tokens:
            if token not in korean_stopwords:
                term_frequency[token] += 1
                term_to_codes[token].add(hs_code)
                korean_terms.add(token)

        # 영문 처리
        english_tokens = tokenize_english(english_name)
        for token in english_tokens:
            if token not in english_stopwords:
                term_frequency[token] += 1
                term_to_codes[token].add(hs_code)
                english_terms.add(token)

    print(f"전체 HS코드: {len(hstable_data):,}")
    print(f"고유 단어 (중복 제거): {len(term_frequency):,}")
    print(f"  - 한글: {len(korean_terms):,}")
    print(f"  - 영문: {len(english_terms):,}")

    return term_frequency, term_to_codes, korean_terms, english_terms


def apply_frequency_threshold(term_frequency, min_freq=3):
    """전략 B: 빈도 임계값 필터링"""
    print("\n" + "="*60)
    print(f"2단계: 빈도 임계값 필터링 (전략 B: min_freq={min_freq})")
    print("="*60)

    # 빈도 분포 분석
    freq_distribution = Counter(term_frequency.values())

    print("\n[빈도 분포]")
    print(f"  1회 등장: {freq_distribution.get(1, 0):,}개")
    print(f"  2회 등장: {freq_distribution.get(2, 0):,}개")
    print(f"  3-5회: {sum(freq_distribution.get(i, 0) for i in range(3, 6)):,}개")
    print(f"  6-10회: {sum(freq_distribution.get(i, 0) for i in range(6, 11)):,}개")
    print(f"  11-20회: {sum(freq_distribution.get(i, 0) for i in range(11, 21)):,}개")
    print(f"  21회 이상: {sum(freq_distribution.get(i, 0) for i in range(21, max(freq_distribution.keys())+1)):,}개")

    # 필터링
    filtered_terms = {term: freq for term, freq in term_frequency.items()
                      if freq >= min_freq}

    print(f"\n[필터링 결과]")
    print(f"  원본 고유 단어: {len(term_frequency):,}")
    print(f"  {min_freq}회 이상 단어: {len(filtered_terms):,}")
    print(f"  제거된 노이즈: {len(term_frequency) - len(filtered_terms):,}")

    return filtered_terms


def calculate_tfidf_scores(hstable_data, min_freq=3):
    """전략 C: TF-IDF 스코어 계산"""
    print("\n" + "="*60)
    print("3단계: TF-IDF 중요도 계산 (전략 C)")
    print("="*60)

    # 각 HS코드를 하나의 문서로 간주
    documents = []
    for item in hstable_data:
        doc = (item.get('한글품명', '') + ' ' + item.get('영문품명', '')).strip()
        documents.append(doc)

    # TF-IDF 계산 (character n-gram으로 한글/영문 모두 처리)
    tfidf = TfidfVectorizer(
        min_df=min_freq,
        analyzer='char',
        ngram_range=(2, 4),
        max_features=50000
    )

    print(f"TF-IDF 벡터화 중... (문서 수: {len(documents):,})")
    tfidf_matrix = tfidf.fit_transform(documents)

    # 각 n-gram의 평균 TF-IDF 스코어
    feature_names = tfidf.get_feature_names_out()
    mean_tfidf_scores = tfidf_matrix.mean(axis=0).A1

    term_importance = dict(zip(feature_names, mean_tfidf_scores))

    # 상위 10개 출력
    sorted_by_importance = sorted(term_importance.items(),
                                  key=lambda x: x[1],
                                  reverse=True)

    print(f"\n[TF-IDF 상위 10개]")
    for term, score in sorted_by_importance[:10]:
        print(f"  '{term}': {score:.6f}")

    return term_importance


def apply_stratified_sampling(filtered_terms, term_to_codes, term_importance,
                               hstable_data, terms_per_chapter=50):
    """전략 D: 계층적 샘플링 (류별 균형)"""
    print("\n" + "="*60)
    print(f"4단계: 계층적 샘플링 (전략 D: 류당 {terms_per_chapter}개)")
    print("="*60)

    # HS코드의 류(Chapter) 추출
    chapter_terms = defaultdict(list)

    for term in filtered_terms:
        if term not in term_to_codes:
            continue

        codes = term_to_codes[term]
        for code in codes:
            # 류 추출 (앞 2자리)
            if len(code) >= 2:
                chapter = code[:2]
                tfidf_score = term_importance.get(term, 0)
                chapter_terms[chapter].append((term, tfidf_score, filtered_terms[term]))

    print(f"\n[계층 정보]")
    print(f"  감지된 류(Chapter) 개수: {len(chapter_terms)}")

    # 각 류별로 TF-IDF 상위 N개 추출
    balanced_terms = {}
    chapter_stats = []

    for chapter in sorted(chapter_terms.keys()):
        terms = chapter_terms[chapter]

        # 중복 제거 (같은 단어가 여러 코드에 나타날 수 있음)
        unique_terms = {}
        for term, tfidf_score, freq in terms:
            if term not in unique_terms:
                unique_terms[term] = (tfidf_score, freq)

        # TF-IDF 기준 정렬
        sorted_terms = sorted(unique_terms.items(),
                             key=lambda x: x[1][0],  # TF-IDF 스코어
                             reverse=True)

        # 상위 N개 선택
        top_n = min(terms_per_chapter, len(sorted_terms))
        for term, (tfidf_score, freq) in sorted_terms[:top_n]:
            balanced_terms[term] = {
                'frequency': freq,
                'tfidf_score': tfidf_score,
                'chapters': list(set(code[:2] for code in term_to_codes[term]))
            }

        chapter_stats.append((chapter, len(unique_terms), top_n))

    # 류별 통계 (샘플 출력)
    print(f"\n[류별 샘플링 통계] (처음 10개 류)")
    for chapter, total, selected in chapter_stats[:10]:
        print(f"  류 {chapter}: {total}개 단어 → {selected}개 선정")

    print(f"\n[최종 결과]")
    print(f"  최종 선정 단어: {len(balanced_terms):,}개")

    return balanced_terms


def calculate_coverage(final_terms, hstable_data):
    """커버리지 분석"""
    print("\n" + "="*60)
    print("5단계: 커버리지 분석")
    print("="*60)

    covered_codes = set()

    for item in hstable_data:
        hs_code = item.get('품목번호', '')
        text = (item.get('한글품명', '') + ' ' + item.get('영문품명', '')).lower()

        # 용어사전의 단어가 하나라도 포함되면 커버
        for term in final_terms:
            if term.lower() in text:
                covered_codes.add(hs_code)
                break

    coverage_rate = len(covered_codes) / len(hstable_data) * 100

    print(f"  커버된 HS코드: {len(covered_codes):,} / {len(hstable_data):,}")
    print(f"  커버율: {coverage_rate:.1f}%")

    return {
        'covered_codes': len(covered_codes),
        'total_codes': len(hstable_data),
        'coverage_rate': coverage_rate
    }


def estimate_size(final_terms):
    """크기 추정"""
    print("\n" + "="*60)
    print("6단계: 크기 추정")
    print("="*60)

    # 간단한 JSON 직렬화 크기 추정
    sample_json = json.dumps(final_terms, ensure_ascii=False)
    json_size_bytes = len(sample_json.encode('utf-8'))
    json_size_kb = json_size_bytes / 1024
    token_estimate = json_size_kb * 300  # 대략 1KB = 300 tokens (한글 포함)

    print(f"  JSON 크기: {json_size_kb:.1f} KB ({json_size_bytes:,} bytes)")
    print(f"  예상 토큰: ~{token_estimate:,.0f} tokens")

    return json_size_kb, token_estimate


def build_terminology(hstable_data, config):
    """통합 용어사전 구축"""
    min_freq = config['min_frequency']
    terms_per_chapter = config['terms_per_chapter']

    # 1단계: 전체 추출
    term_frequency, term_to_codes, korean_terms, english_terms = extract_all_terms(hstable_data)

    # 2단계: 빈도 필터링 (전략 B)
    filtered_terms = apply_frequency_threshold(term_frequency, min_freq)

    # 3단계: TF-IDF 계산 (전략 C)
    term_importance = calculate_tfidf_scores(hstable_data, min_freq)

    # 4단계: 계층 샘플링 (전략 D)
    final_terms = apply_stratified_sampling(filtered_terms, term_to_codes,
                                           term_importance, hstable_data,
                                           terms_per_chapter)

    # 5단계: 커버리지 분석
    coverage = calculate_coverage(final_terms, hstable_data)

    # 6단계: 크기 추정
    json_size_kb, token_estimate = estimate_size(final_terms)

    # 메타데이터 생성
    metadata = {
        'version': config['name'],
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_terms': len(final_terms),
        'korean_terms': sum(1 for t in final_terms if re.match(r'[가-힣]', t)),
        'english_terms': sum(1 for t in final_terms if re.match(r'[a-z]', t)),
        'coverage_rate': round(coverage['coverage_rate'], 1),
        'json_size_kb': round(json_size_kb, 1),
        'estimated_tokens': int(token_estimate),
        'min_frequency': min_freq,
        'terms_per_chapter': terms_per_chapter,
        'optimization': 'B(frequency)+C(tfidf)+D(stratified)'
    }

    # 용어만 추출 (간소화된 버전)
    terms_only = sorted(list(final_terms.keys()))

    return {
        'metadata': metadata,
        'terms': terms_only,
        'term_details': final_terms
    }


def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("HS 품목분류 용어사전 구축 시작")
    print("전략: B(빈도) + C(TF-IDF) + D(계층 샘플링)")
    print("="*60)

    # 데이터 로드
    print("\n데이터 로딩 중...")
    hstable_data = load_hstable()

    # 3가지 버전 설정
    configs = [
        {
            'name': 'minimal',
            'min_frequency': 5,
            'terms_per_chapter': 30,
            'description': '핵심 용어만 (고빈도, 류당 30개)'
        },
        {
            'name': 'balanced',
            'min_frequency': 3,
            'terms_per_chapter': 50,
            'description': '균형잡힌 용어 (권장, 류당 50개)'
        },
        {
            'name': 'full',
            'min_frequency': 2,
            'terms_per_chapter': 100,
            'description': '포괄적 용어 (최대 커버리지, 류당 100개)'
        }
    ]

    # 각 버전 생성
    for config in configs:
        print("\n" + "="*60)
        print(f"버전: {config['name'].upper()} - {config['description']}")
        print("="*60)

        result = build_terminology(hstable_data, config)

        # 파일 저장
        output_file = f"knowledge/hs_terminology_{config['name']}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료: {output_file}")
        print(f"메타데이터: {result['metadata']}")

    print("\n" + "="*60)
    print("모든 버전 생성 완료!")
    print("="*60)
    print("\n생성된 파일:")
    for config in configs:
        print(f"  - knowledge/hs_terminology_{config['name']}.json")


if __name__ == '__main__':
    main()
