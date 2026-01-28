"""
TF-IDF 기반 검색 엔진
Character n-gram 방식 사용 (형태소 분석 불필요)

테스트 결과 Character 기반이 Word 기반보다 우수:
- 복합어 처리 (예: "폴리우레탄폼", "리튬이온배터리")
- 부분 매칭 효과
- 검색 정확도 압도적 우위
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfSearchEngine:
    """
    Character n-gram 기반 TF-IDF 검색 엔진
    - n-gram: (2, 4) - 2~4 글자 조합
    - 형태소 분석 불필요
    - 부분 매칭, 오타 허용에 효과적
    - 복합어 처리 우수
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            min_df=2,              # 최소 2개 문서 이상 등장
            max_df=0.85,           # 85% 이상 문서에 등장한 n-gram 제외 (불용어 조사 필터링)
            max_features=20000,    # 최대 2만 개 n-gram (벡터 차원 제한)
            sublinear_tf=True,     # 로그 스케일 TF
            norm='l2'              # L2 정규화
        )
        self.tfidf_matrix = None
        self.documents = None
        self.doc_ids = None

    def fit(self, documents, doc_ids=None):
        """
        문서 인덱싱

        Args:
            documents: 문서 텍스트 리스트
            doc_ids: 문서 ID 리스트 (선택)
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else list(range(len(documents)))
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(self, query, top_k=10, min_similarity=0.1):
        """
        유사도 기반 검색 (최소 임계값 필터링)

        Args:
            query: 검색 쿼리 (문자열)
            top_k: 반환할 상위 결과 개수
            min_similarity: 최소 유사도 임계값 (0~1, 기본값 0.1)
                          이 값보다 낮은 유사도를 가진 문서는 제외

        Returns:
            [(doc_id, similarity_score), ...]
        """
        if self.tfidf_matrix is None:
            raise ValueError("fit() 메서드를 먼저 호출해야 합니다.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 최소 임계값 이상인 문서만 필터링
        valid_indices = [i for i, score in enumerate(similarities) if score >= min_similarity]

        # 관련 문서가 없으면 빈 리스트 반환
        if not valid_indices:
            return []

        # 필터링된 문서 중 상위 k개 인덱스 추출
        valid_similarities = [(idx, similarities[idx]) for idx in valid_indices]
        valid_similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = valid_similarities[:top_k]

        results = [
            (self.doc_ids[idx], score)
            for idx, score in top_results
        ]

        return results

    def get_similarity_scores(self, query):
        """전체 문서에 대한 유사도 점수 반환"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        return similarities
