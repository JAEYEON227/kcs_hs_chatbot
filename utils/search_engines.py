import json
import re
import time
from typing import List, Dict, Any
from collections import defaultdict
from difflib import SequenceMatcher
from .hs_manual_utils import lookup_hscode
from .text_utils import extract_hs_codes

class TariffTableSearcher:
    def __init__(self):
        self.tariff_data = []
        self.load_tariff_table()

    def load_tariff_table(self):
        """관세율표 데이터 로드"""
        try:
            with open('knowledge/hstable.json', 'r', encoding='utf-8') as f:
                self.tariff_data = json.load(f)
        except FileNotFoundError:
            print("Warning: hstable.json not found")
            self.tariff_data = []

    def calculate_similarity(self, query, text):
        """텍스트 유사도 계산"""
        if not query or not text:
            return 0.0
        return SequenceMatcher(None, query.lower(), text.lower()).ratio()

    def search_by_tariff_table(self, query, top_n=10):
        """관세율표에서 유사도 기반 HS코드 후보 검색"""
        candidates = []

        for item in self.tariff_data:
            hs_code = item.get('품목번호', '')
            korean_name = item.get('한글품명', '')
            english_name = item.get('영문품명', '')

            # 한글품명과 영문품명에서 유사도 계산
            korean_sim = self.calculate_similarity(query, korean_name)
            english_sim = self.calculate_similarity(query, english_name)

            # 최고 유사도 사용
            max_similarity = max(korean_sim, english_sim)

            if max_similarity > 0.1:  # 최소 임계값
                candidates.append({
                    'hs_code': hs_code,
                    'korean_name': korean_name,
                    'english_name': english_name,
                    'similarity': max_similarity,
                    'matched_field': 'korean' if korean_sim > english_sim else 'english'
                })

        # 유사도 순으로 정렬하여 상위 N개 반환
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_n]

class ParallelHSSearcher:
    def __init__(self, hs_manager):
        self.hs_manager = hs_manager
        self.tariff_searcher = TariffTableSearcher()

    def parallel_search(self, query, logger, ui_container=None):
        """병렬적 HS코드 검색"""

        # 경로 1: 관세율표 → 해설서 (2단계)
        logger.log_actual("SEARCH", "Path 1: Tariff Table → Manual search starting...")
        path1_results = self.tariff_to_manual_search(query, logger)

        # 경로 2: 해설서 직접 검색 (기존 방법)
        logger.log_actual("SEARCH", "Path 2: Direct manual search starting...")
        path2_results = self.direct_manual_search(query, logger)

        # 결과 종합
        logger.log_actual("AI", "Consolidating parallel search results...")
        final_results = self.consolidate_results(path1_results, path2_results, logger)

        return final_results

    def tariff_to_manual_search(self, query, logger):
        """경로 1: 관세율표 → 해설서"""
        # 1단계: 관세율표에서 HS코드 후보 선정
        tariff_start = time.time()
        hs_candidates = self.tariff_searcher.search_by_tariff_table(query, top_n=15)
        tariff_time = time.time() - tariff_start

        logger.log_actual("DATA", f"Tariff table search completed",
                         f"{len(hs_candidates)} candidates in {tariff_time:.2f}s")

        if not hs_candidates:
            return []

        # 상위 후보들의 HS코드 리스트 생성
        candidate_codes = [item['hs_code'] for item in hs_candidates[:10]]
        logger.log_actual("INFO", f"Top HS candidates from tariff",
                         f"{', '.join(candidate_codes[:5])}...")

        # 2단계: 해당 HS코드들을 해설서에서 검색
        manual_start = time.time()
        manual_results = []

        for candidate in hs_candidates[:10]:
            hs_code = candidate['hs_code']
            # 해설서에서 해당 HS코드 관련 내용 검색
            manual_content = self.search_manual_by_hs_code(hs_code, query)
            if manual_content:
                manual_results.append({
                    'hs_code': hs_code,
                    'tariff_similarity': candidate['similarity'],
                    'tariff_name': candidate['korean_name'],
                    'manual_content': manual_content,
                    'source': 'tariff_to_manual'
                })

        manual_time = time.time() - manual_start
        logger.log_actual("SUCCESS", f"Manual search for candidates completed",
                         f"{len(manual_results)} results in {manual_time:.2f}s")

        return manual_results

    def search_manual_by_hs_code(self, hs_code, query):
        """특정 HS코드에 대한 해설서 내용 검색"""
        try:
            explanation, type_explanation, number_explanation = lookup_hscode(hs_code, 'knowledge/grouped_11_end.json')

            content = ""
            if explanation and explanation.get('text'):
                content += f"부 해설: {explanation['text']}\n"
            if type_explanation and type_explanation.get('text'):
                content += f"류 해설: {type_explanation['text']}\n"
            if number_explanation and number_explanation.get('text'):
                content += f"호 해설: {number_explanation['text']}\n"

            return content if content else None
        except:
            return None

    def direct_manual_search(self, query, logger):
        """경로 2: 해설서 직접 검색"""
        manual_start = time.time()

        # 해설서 데이터에서 직접 검색
        direct_results = []
        try:
            with open('knowledge/grouped_11_end.json', 'r', encoding='utf-8') as f:
                manual_data = json.load(f)

            # 쿼리 키워드 추출
            query_keywords = self.extract_keywords_from_query(query)

            # 해설서 텍스트에서 매칭되는 항목 찾기
            for item in manual_data:
                text_content = item.get('text', '')
                header1 = item.get('header1', '')
                header2 = item.get('header2', '')

                # 텍스트 내용과 헤더에서 키워드 매칭
                match_score = 0
                full_text = f"{header1} {header2} {text_content}".lower()

                for keyword in query_keywords:
                    if keyword.lower() in full_text:
                        match_score += 1

                if match_score > 0:
                    # HS코드 추출 (header2에서)
                    hs_codes = self.extract_hs_from_header(header2)

                    direct_results.append({
                        'hs_codes': hs_codes,
                        'content': item,
                        'match_score': match_score,
                        'text_content': text_content,
                        'source': 'direct_manual'
                    })

            # 매칭 점수순으로 정렬하여 상위 10개만 선택
            direct_results.sort(key=lambda x: x['match_score'], reverse=True)
            direct_results = direct_results[:10]

        except Exception as e:
            logger.log_actual("ERROR", f"Manual search error: {str(e)}")
            direct_results = []

        manual_time = time.time() - manual_start
        logger.log_actual("SUCCESS", f"Direct manual search completed",
                         f"{len(direct_results)} results in {manual_time:.2f}s")

        return direct_results

    def extract_keywords_from_query(self, query):
        """쿼리에서 키워드 추출"""
        import re
        # 특수문자 제거 및 공백 기준 분리
        words = re.sub(r'[^\w\s]', ' ', query).split()
        # 중복 제거 및 길이 2 이상인 단어만 선택
        return list(set(word for word in words if len(word) >= 2))

    def extract_hs_from_header(self, header):
        """해설서 헤더에서 HS코드 추출"""
        import re
        # "39.11" 형태의 HS코드 패턴 찾기
        hs_pattern = re.findall(r'(\d{2})\.(\d{2})', header)
        if hs_pattern:
            return [f"{code[0]}{code[1]}" for code in hs_pattern]

        # "제39류" 형태에서 류 번호 추출
        chapter_pattern = re.findall(r'제(\d+)류', header)
        if chapter_pattern:
            return [f"{chapter:0>2}00" for chapter in chapter_pattern]

        return []

    def extract_hs_codes_from_content(self, content):
        """해설서 내용에서 HS코드 추출"""
        # 새로운 direct_manual_search 결과 구조에 맞게 수정
        if isinstance(content, dict) and 'hs_codes' in content:
            return content['hs_codes'][:3]  # 최대 3개만
        elif isinstance(content, dict):
            text_content = json.dumps(content, ensure_ascii=False)
        else:
            text_content = str(content)

        # HS코드 패턴 추출
        codes = extract_hs_codes(text_content)
        return codes[:3]  # 최대 3개만

    def consolidate_results(self, path1_results, path2_results, logger):
        """두 경로의 결과를 종합"""
        consolidation_start = time.time()

        # 가중치 설정
        TARIFF_WEIGHT = 0.4  # 관세율표 경로 가중치
        MANUAL_WEIGHT = 0.6  # 해설서 직접 경로 가중치

        final_scores = defaultdict(float)
        result_details = {}

        # 경로 1 결과 처리 (관세율표 → 해설서)
        for result in path1_results:
            hs_code = result['hs_code']
            # 관세율표 유사도 * 가중치
            score = result['tariff_similarity'] * TARIFF_WEIGHT
            final_scores[hs_code] += score

            if hs_code not in result_details:
                result_details[hs_code] = {
                    'hs_code': hs_code,
                    'tariff_name': result.get('tariff_name', ''),
                    'manual_content': result.get('manual_content', ''),
                    'path1_score': score,
                    'path2_score': 0,
                    'sources': ['tariff_to_manual']
                }
            else:
                result_details[hs_code]['sources'].append('tariff_to_manual')

        # 경로 2 결과 처리 (해설서 직접)
        for result in path2_results:
            # HS코드 추출 로직 (해설서 내용에서)
            extracted_codes = self.extract_hs_codes_from_content(result['content'])

            for hs_code in extracted_codes:
                # 해설서 직접 검색 점수 (빈도 기반)
                score = 0.5 * MANUAL_WEIGHT  # 기본 점수
                final_scores[hs_code] += score

                if hs_code not in result_details:
                    result_details[hs_code] = {
                        'hs_code': hs_code,
                        'tariff_name': '',
                        'manual_content': str(result['content']),
                        'path1_score': 0,
                        'path2_score': score,
                        'sources': ['direct_manual']
                    }
                else:
                    result_details[hs_code]['path2_score'] += score
                    if 'direct_manual' not in result_details[hs_code]['sources']:
                        result_details[hs_code]['sources'].append('direct_manual')

        # 최종 순위 정렬
        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        consolidation_time = time.time() - consolidation_start
        logger.log_actual("SUCCESS", f"Results consolidation completed",
                         f"{len(sorted_results)} unique HS codes in {consolidation_time:.2f}s")

        # 상위 2개 결과 반환
        top_results = []
        for hs_code, final_score in sorted_results[:2]:
            if hs_code in result_details:
                details = result_details[hs_code]
                details['final_score'] = final_score
                details['confidence'] = 'HIGH' if len(details['sources']) > 1 else 'MEDIUM'
                top_results.append(details)

        return top_results

    def create_enhanced_context(self, search_results):
        """검색 결과를 컨텍스트로 변환"""
        context = ""

        for i, result in enumerate(search_results, 1):
            context += f"\n=== 후보 {i}: HS코드 {result['hs_code']} ===\n"
            context += f"신뢰도: {result['confidence']}\n"
            context += f"최종점수: {result['final_score']:.3f}\n"

            if result['tariff_name']:
                context += f"관세율표 품목명: {result['tariff_name']}\n"

            context += f"검색경로: {', '.join(result['sources'])}\n"

            if result.get('manual_summary'):
                context += f"해설서 요약:\n{result['manual_summary']}\n"
            elif result['manual_content']:
                context += f"해설서 내용:\n{result['manual_content'][:1000]}...\n"

            context += "\n"

        return context