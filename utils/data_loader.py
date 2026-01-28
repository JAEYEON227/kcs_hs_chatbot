"""
HS 데이터 로더 (리팩토링 버전)

이 모듈은 HS 분류 사례 데이터를 로드하고 관리하는 기본 기능만 제공합니다.
검색 기능은 별도의 searcher 클래스들로 분리되었습니다.
"""

import json
from typing import Dict, List, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class HSDataManager:
    """
    HS 코드 관련 데이터를 관리하는 클래스

    역할:
    - HS 분류 사례, 위원회 결정, 협의회 결정 등의 데이터를 로드하고 관리
    - 데이터 접근 인터페이스 제공

    검색 기능은 다음 클래스들로 분리:
    - TfidfCaseSearcher: TF-IDF 기반 검색
    - KeywordCaseSearcher: 키워드 기반 검색
    """

    def __init__(self):
        """HSDataManager 초기화"""
        self.data = {}  # 모든 HS 관련 데이터를 저장하는 딕셔너리
        self.load_all_data()  # 모든 데이터 파일 로드

    def load_all_data(self):
        """
        모든 HS 데이터 파일을 로드하는 메서드
        - HS분류사례_part1~10.json 파일 로드
        - HS위원회.json, HS협의회.json 파일 로드
        - hs_classification_data_us.json 파일 로드 (미국 관세청 품목분류 사례)
        - hs_classification_data_eu.json 파일 로드 (EU 관세청 품목분류 사례)
        """
        # HS분류사례 파트 로드 (1~10)
        for i in range(1, 11):
            try:
                with open(f'knowledge/HS분류사례_part{i}.json', 'r', encoding='utf-8') as f:
                    self.data[f'HS분류사례_part{i}'] = json.load(f)
            except FileNotFoundError:
                print(f'Warning: HS분류사례_part{i}.json not found')

        # 기타 JSON 파일 로드 (위원회, 협의회 결정)
        other_files = ['knowledge/HS위원회.json', 'knowledge/HS협의회.json']
        for file in other_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    self.data[file.replace('.json', '')] = json.load(f)
            except FileNotFoundError:
                print(f'Warning: {file} not found')

        # 미국 관세청 품목분류 사례 로드
        try:
            with open('knowledge/hs_classification_data_us.json', 'r', encoding='utf-8') as f:
                self.data['hs_classification_data_us'] = json.load(f)
        except FileNotFoundError:
            print('Warning: hs_classification_data_us.json not found')

        # EU 관세청 품목분류 사례 로드
        try:
            with open('knowledge/hs_classification_data_eu.json', 'r', encoding='utf-8') as f:
                self.data['hs_classification_data_eu'] = json.load(f)
        except FileNotFoundError:
            print('Warning: hs_classification_data_eu.json not found')

    def get_domestic_data(self) -> Dict[str, List[Any]]:
        """
        국내 HS 분류 사례 데이터 반환

        Returns:
            국내 데이터 딕셔너리 (키: 소스명, 값: 사례 리스트)
        """
        domestic_keys = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]
        return {key: self.data[key] for key in domestic_keys if key in self.data}

    def get_overseas_data(self) -> Dict[str, List[Any]]:
        """
        해외 HS 분류 사례 데이터 반환

        Returns:
            해외 데이터 딕셔너리 (키: 소스명, 값: 사례 리스트)
        """
        overseas_keys = ['hs_classification_data_us', 'hs_classification_data_eu']
        return {key: self.data[key] for key in overseas_keys if key in self.data}

    def get_all_data(self) -> Dict[str, List[Any]]:
        """
        모든 데이터 반환

        Returns:
            전체 데이터 딕셔너리
        """
        return self.data
