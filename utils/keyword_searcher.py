"""
키워드 기반 HS 분류 사례 검색 엔진

이 모듈은 단순 키워드 매칭 방식으로 HS 분류 사례를 검색합니다.
- 참고문서번호 검색
- 키워드 토큰 기반 OR 검색
- HS 코드 직접 검색
"""

import re
from typing import List, Dict, Any


class KeywordCaseSearcher:
    """
    키워드 기반 HS 분류 사례 검색 클래스

    TF-IDF 검색과 달리 단순 문자열 매칭으로 사례를 검색합니다.
    참고문서번호 검색이나 특정 키워드 정확 매칭에 유용합니다.
    """

    def __init__(self, data_manager):
        """
        Args:
            data_manager: HSDataManager 인스턴스 (데이터 소스)
        """
        self.data_manager = data_manager

    def _tokenize_query(self, query: str) -> List[str]:
        """
        검색 쿼리를 토큰으로 분리하는 메서드 (검색 전용)

        Args:
            query: 검색 쿼리 문자열

        Returns:
            토큰 리스트 (최소 2글자 이상)
        """
        # 특수문자 제거 및 공백 기준 분리
        tokens = re.sub(r'[^\w\s]', ' ', query).split()
        # 길이 2 이상인 토큰만 반환 (중복 허용 - 빈도 계산에 사용)
        return [token.strip() for token in tokens if len(token.strip()) >= 2]

    def search_domestic_by_keyword(
        self,
        keyword: str,
        top_k: int = 10,
        ignore_spaces: bool = False,
        min_tokens: int = 1
    ) -> List[Dict[str, Any]]:
        """
        고급 키워드 검색 (국내 분류사례)

        개선 사항:
        1. 토큰화: 검색어를 공백으로 분리
        2. OR 검색: 토큰 중 하나라도 매칭
        3. 띄어쓰기 무시: 공백 제거 후 검색 옵션
        4. 가중치 적용: 매칭된 토큰 개수로 순위 매김

        Args:
            keyword: 검색할 키워드 (여러 단어 가능)
            top_k: 반환할 최대 결과 개수
            ignore_spaces: True시 띄어쓰기 무시하고 검색
            min_tokens: 최소 매칭 토큰 수 (기본 1)

        Returns:
            검색 결과 리스트 (가중치 순으로 정렬)
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        # 1. 토큰화
        tokens = self._tokenize_query(keyword)
        if not tokens:
            return []

        tokens_lower = [t.lower() for t in tokens]

        # 결과를 점수와 함께 저장
        scored_results = []

        for source in domestic_sources:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    # 품목명, 설명, 분류근거에서 검색
                    searchable_text = ' '.join([
                        str(item.get('product_name', '')),
                        str(item.get('description', '')),
                        str(item.get('decision_reason', ''))
                    ]).lower()

                    # 띄어쓰기 무시 옵션
                    if ignore_spaces:
                        searchable_text_no_space = searchable_text.replace(' ', '')

                    # 가중치 계산: 매칭된 토큰 개수 카운트
                    matched_tokens = 0
                    for token in tokens_lower:
                        if ignore_spaces:
                            token_no_space = token.replace(' ', '')
                            if token_no_space in searchable_text_no_space:
                                matched_tokens += 1
                        else:
                            if token in searchable_text:
                                matched_tokens += 1

                    # OR 검색: 최소 토큰 수 이상 매칭되면 포함
                    if matched_tokens >= min_tokens:
                        scored_results.append((matched_tokens, item))

        # 점수 기준 내림차순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # 상위 top_k개만 반환 (점수는 제외)
        return [item for score, item in scored_results[:top_k]]

    def find_domestic_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 국내 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "품목분류2과-9433")

        Returns:
            해당 사례 딕셔너리 또는 None
        """
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]

        for source in domestic_sources:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    if item.get('reference_id') == ref_id:
                        return item
        return None

    def search_overseas_by_keyword(
        self,
        keyword: str,
        top_k: int = 10,
        ignore_spaces: bool = False,
        min_tokens: int = 1
    ) -> List[Dict[str, Any]]:
        """
        고급 키워드 검색 (해외 분류사례)

        개선 사항:
        1. 토큰화: 검색어를 공백으로 분리
        2. OR 검색: 토큰 중 하나라도 매칭
        3. 띄어쓰기 무시: 공백 제거 후 검색 옵션
        4. 가중치 적용: 매칭된 토큰 개수로 순위 매김

        Args:
            keyword: 검색할 키워드 (여러 단어 가능)
            top_k: 반환할 최대 결과 개수
            ignore_spaces: True시 띄어쓰기 무시하고 검색
            min_tokens: 최소 매칭 토큰 수 (기본 1)

        Returns:
            검색 결과 리스트 (가중치 순으로 정렬)
        """
        # 1. 토큰화
        tokens = self._tokenize_query(keyword)
        if not tokens:
            return []

        tokens_lower = [t.lower() for t in tokens]

        # 결과를 점수와 함께 저장
        scored_results = []

        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    # 품목명, 설명, reply에서 검색
                    searchable_text = ' '.join([
                        str(item.get('product_name', '')),
                        str(item.get('description', '')),
                        str(item.get('reply', ''))
                    ]).lower()

                    # 띄어쓰기 무시 옵션
                    if ignore_spaces:
                        searchable_text_no_space = searchable_text.replace(' ', '')

                    # 가중치 계산: 매칭된 토큰 개수 카운트
                    matched_tokens = 0
                    for token in tokens_lower:
                        if ignore_spaces:
                            token_no_space = token.replace(' ', '')
                            if token_no_space in searchable_text_no_space:
                                matched_tokens += 1
                        else:
                            if token in searchable_text:
                                matched_tokens += 1

                    # OR 검색: 최소 토큰 수 이상 매칭되면 포함
                    if matched_tokens >= min_tokens:
                        scored_results.append((matched_tokens, item))

        # 점수 기준 내림차순 정렬
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # 상위 top_k개만 반환 (점수는 제외)
        return [item for score, item in scored_results[:top_k]]

    def find_overseas_case_by_id(self, ref_id: str) -> Dict[str, Any]:
        """
        참고문서번호로 해외 분류사례 검색

        Args:
            ref_id: 참고문서번호 (예: "NY N338825")

        Returns:
            {'case': 사례 딕셔너리, 'country': 'US'/'EU'} 또는 None
        """
        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                for item in self.data_manager.data[source]:
                    if item.get('reference_id') == ref_id:
                        country = 'US' if 'us' in source else 'EU'
                        return {'case': item, 'country': country}
        return None

    def search_overseas_by_hs_code(self, hs_code: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        HS 코드로 해외 분류사례 검색

        Args:
            hs_code: HS 코드 (4-10자리, 예: "5515", "5515.12")
            top_k: 반환할 최대 결과 개수

        Returns:
            매칭되는 사례 리스트 (국가 정보 포함)
        """
        results = []

        for source in ['hs_classification_data_us', 'hs_classification_data_eu']:
            if source in self.data_manager.data:
                country = 'US' if 'us' in source else 'EU'
                for item in self.data_manager.data[source]:
                    item_hs_code = item.get('hs_code', '')
                    # HS 코드 부분 매칭 (공백, 점, 하이픈 제거 후 비교)
                    if hs_code.replace('.', '').replace(' ', '') in item_hs_code.replace('.', '').replace(' ', '').replace('-', ''):
                        results.append({
                            'case': item,
                            'country': country
                        })
                        if len(results) >= top_k:
                            return results
        return results
