"""
TF-IDF 검색 인덱스 사전 구축 스크립트

사용법:
    python build_tfidf_index.py

출력:
    - tfidf_indexes.pkl.gz: 국내/해외 TF-IDF 인덱스 (gzip 압축)

주의:
    - knowledge/ 폴더에 데이터 파일들이 있어야 함
    - 데이터 변경 시마다 재실행 필요
"""

import pickle
import gzip
import os
import time
from utils import HSDataManager

def main():
    print("=" * 60)
    print("TF-IDF 검색 인덱스 구축 시작")
    print("=" * 60)

    start_time = time.time()

    # HSDataManager 초기화 (자동으로 TF-IDF 인덱스 구축)
    print("\n1. 데이터 로딩 및 TF-IDF 인덱스 구축 중...")
    hs_manager = HSDataManager()
    build_time = time.time() - start_time
    print(f"   완료! (소요 시간: {build_time:.2f}초)")

    # 인덱스 정보 출력
    print("\n2. 구축된 인덱스 정보:")
    if hs_manager.domestic_tfidf:
        print(f"   - 국내 사례: {len(hs_manager.domestic_items)}개")
    if hs_manager.overseas_tfidf:
        print(f"   - 해외 사례: {len(hs_manager.overseas_items)}개")

    # gzip 압축 pickle 파일로 저장
    print("\n3. 인덱스 파일 저장 중...")
    save_start = time.time()

    with gzip.open('tfidf_indexes.pkl.gz', 'wb', compresslevel=9) as f:
        pickle.dump({
            'domestic_tfidf': hs_manager.domestic_tfidf,
            'domestic_items': hs_manager.domestic_items,
            'overseas_tfidf': hs_manager.overseas_tfidf,
            'overseas_items': hs_manager.overseas_items
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    save_time = time.time() - save_start
    size_mb = os.path.getsize('tfidf_indexes.pkl.gz') / (1024 * 1024)
    print(f"   저장 완료! (소요 시간: {save_time:.2f}초)")
    print(f"   파일 크기: {size_mb:.2f} MB")
    print(f"   파일 위치: tfidf_indexes.pkl.gz")

    # 전체 소요 시간
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"전체 작업 완료! (총 소요 시간: {total_time:.2f}초)")
    print("=" * 60)
    print("\nStreamlit 실행 시 이 인덱스 파일을 자동으로 로드합니다.")
    print("데이터 변경 시 이 스크립트를 다시 실행하세요.")

if __name__ == "__main__":
    main()
