"""
TF-IDF 기반 HS 분류 사례 검색 엔진

이 모듈은 국내 및 해외 HS 분류 사례에 대한 TF-IDF 기반 검색 기능을 제공합니다.
- Character n-gram 방식 사용 (형태소 분석 불필요)
- pickle 파일로 인덱스 저장/로드하여 빠른 초기화
"""

import os
import pickle
import gzip
from typing import List, Dict, Any
from .tfidf_search import TfidfSearchEngine


class TfidfCaseSearcher:
    """
    TF-IDF 기반 HS 분류 사례 검색 클래스

    국내 사례와 해외 사례에 대해 각각 독립적인 TF-IDF 인덱스를 구축하고
    검색 기능을 제공합니다.
    """

    def __init__(self, data_manager):
        """
        Args:
            data_manager: HSDataManager 인스턴스 (데이터 소스)
        """
        self.data_manager = data_manager
        self.domestic_tfidf = None
        self.overseas_tfidf = None
        self.domestic_items = []
        self.overseas_items = []

        # TF-IDF 인덱스: pickle 파일이 있으면 로드, 없으면 구축
        if os.path.exists('tfidf_indexes.pkl.gz') or os.path.exists('tfidf_indexes.pkl'):
            self._load_indexes()
        else:
            self.build_indexes()

    def _load_indexes(self):
        """
        사전에 구축된 TF-IDF 인덱스를 pickle 파일에서 로드
        """
        try:
            # gzip 압축 파일 우선 시도
            if os.path.exists('tfidf_indexes.pkl.gz'):
                with gzip.open('tfidf_indexes.pkl.gz', 'rb') as f:
                    indexes = pickle.load(f)
            # 기존 비압축 파일 호환성 유지
            elif os.path.exists('tfidf_indexes.pkl'):
                with open('tfidf_indexes.pkl', 'rb') as f:
                    indexes = pickle.load(f)
            else:
                raise FileNotFoundError("No index file found")

            self.domestic_tfidf = indexes['domestic_tfidf']
            self.domestic_items = indexes['domestic_items']
            self.overseas_tfidf = indexes['overseas_tfidf']
            self.overseas_items = indexes['overseas_items']
        except Exception as e:
            print(f"Warning: Failed to load TF-IDF indexes: {e}")
            print("Building indexes from scratch...")
            self.build_indexes()

    def build_indexes(self):
        """
        TF-IDF 검색 인덱스 구축
        - 국내 사례, 해외 사례 각각 별도 인덱스 구축
        - 구축 후 gzip 압축하여 pickle 파일로 저장
        """
        # 1. 국내 HS 분류 사례 인덱스
        domestic_docs = []
        domestic_items = []

        for i in range(1, 11):
            key = f'HS분류사례_part{i}'
            if key in self.data_manager.data:
                for item in self.data_manager.data[key]:
                    # 문서 텍스트 생성
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        domestic_docs.append(text)
                        domestic_items.append(item)

        # 위원회, 협의회 데이터 추가
        for key in ['knowledge/HS위원회', 'knowledge/HS협의회']:
            if key in self.data_manager.data:
                for item in self.data_manager.data[key]:
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        domestic_docs.append(text)
                        domestic_items.append(item)

        if domestic_docs:
            self.domestic_tfidf = TfidfSearchEngine()
            self.domestic_tfidf.fit(domestic_docs)
            self.domestic_items = domestic_items

        # 2. 해외 HS 분류 사례 인덱스
        overseas_docs = []
        overseas_items = []

        for key in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if key in self.data_manager.data:
                for item in self.data_manager.data[key]:
                    text_parts = []
                    if 'product_name' in item and item['product_name']:
                        text_parts.append(item['product_name'])
                    if 'description' in item and item['description']:
                        text_parts.append(item['description'])
                    if 'decision_reason' in item and item['decision_reason']:
                        text_parts.append(item['decision_reason'])

                    text = " ".join(text_parts)
                    if text.strip():
                        overseas_docs.append(text)
                        overseas_items.append(item)

        if overseas_docs:
            self.overseas_tfidf = TfidfSearchEngine()
            self.overseas_tfidf.fit(overseas_docs)
            self.overseas_items = overseas_items

        # 3. 구축한 인덱스를 gzip 압축하여 pickle 파일로 저장
        try:
            with gzip.open('tfidf_indexes.pkl.gz', 'wb', compresslevel=9) as f:
                pickle.dump({
                    'domestic_tfidf': self.domestic_tfidf,
                    'domestic_items': self.domestic_items,
                    'overseas_tfidf': self.overseas_tfidf,
                    'overseas_items': self.overseas_items
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 파일 크기 정보 출력
            size_mb = os.path.getsize('tfidf_indexes.pkl.gz') / (1024 * 1024)
            print(f"TF-IDF 인덱스가 tfidf_indexes.pkl.gz 파일로 저장되었습니다.")
            print(f"   파일 크기: {size_mb:.2f} MB")
        except Exception as e:
            print(f"TF-IDF 인덱스 저장 실패: {e}")

    def search_domestic(self, query: str, top_k: int = 100, min_similarity: float = 0.1,
                       expanded_query: str = None) -> List[Dict[str, Any]]:
        """
        TF-IDF 기반 국내 HS 분류 사례 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            min_similarity: 최소 유사도 임계값 (기본값 0.1)
            expanded_query: AI가 확장한 쿼리 (선택사항)

        Returns:
            검색된 항목 리스트
        """
        if self.domestic_tfidf is None:
            return []

        # 확장된 쿼리가 제공되면 우선 사용
        search_query = expanded_query if expanded_query else query

        results = self.domestic_tfidf.search(search_query, top_k, min_similarity)
        return [self.domestic_items[idx] for idx, score in results]

    def search_overseas(self, query: str, top_k: int = 100, min_similarity: float = 0.1,
                       expanded_query: str = None) -> List[Dict[str, Any]]:
        """
        TF-IDF 기반 해외 HS 분류 사례 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            min_similarity: 최소 유사도 임계값 (기본값 0.1)
            expanded_query: AI가 확장한 쿼리 (선택사항)

        Returns:
            검색된 항목 리스트
        """
        if self.overseas_tfidf is None:
            return []

        # 확장된 쿼리가 제공되면 우선 사용
        search_query = expanded_query if expanded_query else query

        results = self.overseas_tfidf.search(search_query, top_k, min_similarity)
        return [self.overseas_items[idx] for idx, score in results]
