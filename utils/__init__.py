"""
utils package - HS 품목분류 챗봇 유틸리티 모듈 (리팩토링 버전)

이 패키지는 기능별로 분할된 모듈들로 구성되어 있습니다:
- data_loader: HS 데이터 로딩 전용
- tfidf_case_searcher: TF-IDF 기반 검색
- keyword_searcher: 키워드 기반 검색
- text_utils: 텍스트 처리 유틸리티
- question_classifier: LLM 기반 질문 분류
- hs_manual_utils: HS 해설서 관련 함수들
- search_engines: 품목분류표 및 해설서 검색 엔진
- handlers: 질문 유형별 처리 함수들
- api_retry: API 재시도 로직 (503/429 에러 자동 복구)

backward compatibility를 위해 Facade 패턴으로 기존 인터페이스 유지
"""

# Core data components
from .data_loader import HSDataManager as _HSDataManager
from .tfidf_case_searcher import TfidfCaseSearcher
from .keyword_searcher import KeywordCaseSearcher

# Search engines (키워드 기반 유지)
from .search_engines import TariffTableSearcher, ParallelHSSearcher

# Text utilities
from .text_utils import clean_text, extract_hs_codes

# HS manual utilities
from .hs_manual_utils import (
    lookup_hscode,
    get_hs_explanations,
    get_tariff_info_for_codes,
    get_manual_info_for_codes,
    prepare_general_rules,
    analyze_user_provided_codes
)

# Question classifier
from .question_classifier import classify_question

# Handlers
from .handlers import (
    handle_web_search,
    handle_hs_classification_cases,
    handle_overseas_hs,
    handle_hs_manual_with_user_codes,
    handle_domestic_case_lookup,
    handle_overseas_case_lookup
)

# API retry utilities
from .api_retry import (
    retry_on_api_error,
    retry_api_call,
    create_retry_callback_for_streamlit
)


# ==================== Facade 패턴: 하위 호환성 유지 ====================

class HSDataManager:
    """
    HSDataManager Facade 클래스

    기존 코드와의 호환성을 위해 TfidfCaseSearcher와 KeywordCaseSearcher의
    메서드들을 위임(delegation)하는 Facade 패턴을 사용합니다.

    내부적으로는 3개의 클래스로 분리되어 있지만,
    외부에서는 기존처럼 단일 클래스로 사용 가능합니다.
    """

    def __init__(self):
        # 실제 데이터 로더
        self._data_loader = _HSDataManager()

        # TF-IDF 검색 엔진
        self.tfidf_searcher = TfidfCaseSearcher(self._data_loader)

        # 키워드 검색 엔진
        self.keyword_searcher = KeywordCaseSearcher(self._data_loader)

        # 하위 호환성을 위한 속성 노출
        self.data = self._data_loader.data
        self.domestic_tfidf = self.tfidf_searcher.domestic_tfidf
        self.overseas_tfidf = self.tfidf_searcher.overseas_tfidf
        self.domestic_items = self.tfidf_searcher.domestic_items
        self.overseas_items = self.tfidf_searcher.overseas_items

    # ==================== TF-IDF 검색 메서드 (위임) ====================

    def search_domestic_tfidf(self, *args, **kwargs):
        """TF-IDF 기반 국내 사례 검색 (TfidfCaseSearcher로 위임)"""
        return self.tfidf_searcher.search_domestic(*args, **kwargs)

    def search_overseas_tfidf(self, *args, **kwargs):
        """TF-IDF 기반 해외 사례 검색 (TfidfCaseSearcher로 위임)"""
        return self.tfidf_searcher.search_overseas(*args, **kwargs)

    def build_tfidf_indexes(self):
        """TF-IDF 인덱스 구축 (TfidfCaseSearcher로 위임)"""
        return self.tfidf_searcher.build_indexes()

    # ==================== 키워드 검색 메서드 (위임) ====================

    def search_domestic_by_keyword(self, *args, **kwargs):
        """키워드 기반 국내 사례 검색 (KeywordCaseSearcher로 위임)"""
        return self.keyword_searcher.search_domestic_by_keyword(*args, **kwargs)

    def search_overseas_by_keyword(self, *args, **kwargs):
        """키워드 기반 해외 사례 검색 (KeywordCaseSearcher로 위임)"""
        return self.keyword_searcher.search_overseas_by_keyword(*args, **kwargs)

    def find_domestic_case_by_id(self, *args, **kwargs):
        """참고문서번호로 국내 사례 검색 (KeywordCaseSearcher로 위임)"""
        return self.keyword_searcher.find_domestic_case_by_id(*args, **kwargs)

    def find_overseas_case_by_id(self, *args, **kwargs):
        """참고문서번호로 해외 사례 검색 (KeywordCaseSearcher로 위임)"""
        return self.keyword_searcher.find_overseas_case_by_id(*args, **kwargs)

    def search_overseas_by_hs_code(self, *args, **kwargs):
        """HS 코드로 해외 사례 검색 (KeywordCaseSearcher로 위임)"""
        return self.keyword_searcher.search_overseas_by_hs_code(*args, **kwargs)

    # ==================== 데이터 접근 메서드 ====================

    def get_domestic_data(self):
        """국내 데이터 반환"""
        return self._data_loader.get_domestic_data()

    def get_overseas_data(self):
        """해외 데이터 반환"""
        return self._data_loader.get_overseas_data()

    def get_all_data(self):
        """전체 데이터 반환"""
        return self._data_loader.get_all_data()


# ==================== Export 목록 ====================

__all__ = [
    # Classes
    'HSDataManager',
    'TfidfCaseSearcher',
    'KeywordCaseSearcher',
    'TariffTableSearcher',
    'ParallelHSSearcher',

    # Text utils
    'clean_text',
    'extract_hs_codes',

    # HS manual utils
    'lookup_hscode',
    'get_hs_explanations',
    'get_tariff_info_for_codes',
    'get_manual_info_for_codes',
    'prepare_general_rules',
    'analyze_user_provided_codes',

    # Classification
    'classify_question',

    # Handlers
    'handle_web_search',
    'handle_hs_classification_cases',
    'handle_overseas_hs',
    'handle_hs_manual_with_user_codes',
    'handle_domestic_case_lookup',
    'handle_overseas_case_lookup',

    # API retry utilities
    'retry_on_api_error',
    'retry_api_call',
    'create_retry_callback_for_streamlit'
]

__version__ = '3.0.0'  # 리팩토링 버전
__author__ = 'HS Chatbot Team'
