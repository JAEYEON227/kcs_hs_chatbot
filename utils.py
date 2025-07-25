import json
import re
import os
import requests
from typing import Dict, List, Any
from collections import defaultdict
from google import genai
from google.genai import types
from dotenv import load_dotenv



# 환경 변수 로드 (.env 파일에서 API 키 등 설정값 로드)
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

class HSDataManager:
    """
    HS 코드 관련 데이터를 관리하는 클래스
    - HS 분류 사례, 위원회 결정, 협의회 결정 등의 데이터를 로드하고 관리
    - 키워드 기반 검색 기능 제공
    - 관련 컨텍스트 생성 기능 제공
    """
    
    def __init__(self):
        """HSDataManager 초기화"""
        self.data = {}  # 모든 HS 관련 데이터를 저장하는 딕셔너리
        self.search_index = defaultdict(list)  # 키워드 기반 검색을 위한 인덱스
        self.load_all_data()  # 모든 데이터 파일 로드
        self.build_search_index()  # 검색 인덱스 구축
    
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
    
    def build_search_index(self):
        """
        검색 인덱스 구축 메서드
        - 각 데이터 항목에서 키워드를 추출
        - 추출된 키워드를 인덱스에 저장하여 빠른 검색 가능
        """
        for source, items in self.data.items():
            for item in items:
                # 품목명에서 키워드 추출
                keywords = self._extract_keywords(str(item))
                # 각 키워드에 대해 해당 아이템 참조 저장
                for keyword in keywords:
                    self.search_index[keyword].append((source, item))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 의미있는 키워드를 추출하는 내부 메서드
        Args:
            text: 키워드를 추출할 텍스트
        Returns:
            추출된 키워드 리스트
        """
        # 특수문자 제거 및 공백 기준 분리
        words = re.sub(r'[^\w\s]', ' ', text).split()
        # 중복 제거 및 길이 2 이상인 단어만 선택
        return list(set(word for word in words if len(word) >= 2))
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        쿼리와 관련된 가장 연관성 높은 항목들을 검색하는 메서드
        Args:
            query: 검색할 쿼리 문자열
            max_results: 반환할 최대 결과 수 (기본값: 5)
        Returns:
            검색 결과 리스트 (출처와 항목 정보 포함)
        """
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)
        
        # 각 키워드에 대해 매칭되는 항목 찾기
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                # 가중치 계산 (키워드 매칭 횟수 기반)
                results[(source, str(item))] += 1
        
        # 가중치 기준 정렬
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 결과만 반환
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]
    
    def search_domestic_group(self, query: str, group_idx: int, max_results: int = 3) -> List[Dict[str, Any]]:
        """국내 HS 분류 데이터 그룹별 검색 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)

        # 그룹별 데이터 소스 정의 (5개 그룹)
        group_sources = [
            ['HS분류사례_part1', 'HS분류사례_part2'],  # 그룹1
            ['HS분류사례_part3', 'HS분류사례_part4'],  # 그룹2
            ['HS분류사례_part5', 'HS분류사례_part6'],  # 그룹3
            ['HS분류사례_part7', 'HS분류사례_part8'],  # 그룹4
            ['HS분류사례_part9', 'HS분류사례_part10', 'knowledge/HS위원회', 'knowledge/HS협의회']  # 그룹5
        ]
        sources = group_sources[group_idx]

        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                if source in sources:
                    results[(source, str(item))] += 1

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def get_domestic_context_group(self, query: str, group_idx: int) -> str:
        """국내 HS 분류 관련 컨텍스트(그룹별)를 생성하는 메서드"""
        results = self.search_domestic_group(query, group_idx)
        context = []
        for result in results:
            context.append(f"출처: {result['source']} (국내 관세청)\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        return "\n\n".join(context)

    def search_overseas_group(self, query: str, group_idx: int, max_results: int = 3) -> List[Dict[str, Any]]:
        """해외 HS 분류 데이터 그룹별 검색 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)
        
        # 해외 데이터를 그룹별로 분할 처리
        if group_idx < 3:  # 그룹 0,1,2는 미국 데이터
            target_source = 'hs_classification_data_us'
            # 미국 데이터를 3등분
            us_data = self.data.get(target_source, [])
            chunk_size = len(us_data) // 3
            start_idx = group_idx * chunk_size
            end_idx = start_idx + chunk_size if group_idx < 2 else len(us_data)
            target_items = us_data[start_idx:end_idx]
        else:  # 그룹 3,4는 EU 데이터
            target_source = 'hs_classification_data_eu'
            # EU 데이터를 2등분
            eu_data = self.data.get(target_source, [])
            chunk_size = len(eu_data) // 2
            eu_group_idx = group_idx - 3  # 0 or 1
            start_idx = eu_group_idx * chunk_size
            end_idx = start_idx + chunk_size if eu_group_idx < 1 else len(eu_data)
            target_items = eu_data[start_idx:end_idx]
        
        # 해당 그룹 데이터에서만 검색
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                if source == target_source and item in target_items:
                    results[(source, str(item))] += 1
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]

    def get_overseas_context_group(self, query: str, group_idx: int) -> str:
        """해외 HS 분류 관련 컨텍스트(그룹별)를 생성하는 메서드"""
        results = self.search_overseas_group(query, group_idx)
        context = []
        
        for result in results:
            # 출처에 따라 국가 구분
            if result['source'] == 'hs_classification_data_us':
                country = "미국 관세청"
            elif result['source'] == 'hs_classification_data_eu':
                country = "EU 관세청"
            else:
                country = "해외 관세청"
                
            context.append(f"출처: {result['source']} ({country})\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        
        return "\n\n".join(context)
    
    def search_domestic(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """국내 HS 분류 데이터에서만 검색하는 메서드"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)
        
        # 국내 데이터 소스만 필터링
        domestic_sources = [
            'HS분류사례_part1', 'HS분류사례_part2', 'HS분류사례_part3', 'HS분류사례_part4', 'HS분류사례_part5',
            'HS분류사례_part6', 'HS분류사례_part7', 'HS분류사례_part8', 'HS분류사례_part9', 'HS분류사례_part10',
            'knowledge/HS위원회', 'knowledge/HS협의회'
        ]
        
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                # 국내 데이터 소스만 포함
                if source in domestic_sources:
                    results[(source, str(item))] += 1
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]
    
    def get_domestic_context(self, query: str) -> str:
        """국내 HS 분류 관련 컨텍스트를 생성하는 메서드"""
        results = self.search_domestic(query)
        context = []
        
        for result in results:
            context.append(f"출처: {result['source']} (국내 관세청)\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        
        return "\n\n".join(context)
    
    def search_overseas_improved(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """해외 HS 분류 데이터에서만 검색하는 개선된 메서드 (search_index 활용)"""
        query_keywords = self._extract_keywords(query)
        results = defaultdict(int)
        
        # 해외 데이터 소스만 필터링
        overseas_sources = ['hs_classification_data_us', 'hs_classification_data_eu']
        
        for keyword in query_keywords:
            for source, item in self.search_index.get(keyword, []):
                # 해외 데이터 소스만 포함
                if source in overseas_sources:
                    results[(source, str(item))] += 1
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'source': source, 'item': eval(item_str)}
            for (source, item_str), _ in sorted_results[:max_results]
        ]
    
    def get_domestic_context(self, query: str) -> str:
        """국내 HS 분류 관련 컨텍스트를 생성하는 메서드"""
        results = self.search_domestic(query)
        context = []
        
        for result in results:
            context.append(f"출처: {result['source']} (국내 관세청)\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        
        return "\n\n".join(context)


    def get_relevant_context(self, query: str) -> str:
        """
        쿼리에 관련된 컨텍스트를 생성하는 메서드
        Args:
            query: 컨텍스트를 생성할 쿼리 문자열
        Returns:
            관련 컨텍스트 문자열 (출처와 항목 정보 포함)
        """
        results = self.search(query)
        context = []
        
        for result in results:
            context.append(f"출처: {result['source']}\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        
        return "\n\n".join(context)
    
    def get_overseas_context_improved(self, query: str) -> str:
        """해외 HS 분류 관련 컨텍스트를 생성하는 개선된 메서드"""
        results = self.search_overseas_improved(query)
        context = []
        
        for result in results:
            # 출처에 따라 국가 구분
            if result['source'] == 'hs_classification_data_us':
                country = "미국 관세청"
            elif result['source'] == 'hs_classification_data_eu':
                country = "EU 관세청"
            else:
                country = "해외 관세청"
                
            context.append(f"출처: {result['source']} ({country})\n항목: {json.dumps(result['item'], ensure_ascii=False)}")
        
        return "\n\n".join(context)

# HTML 태그 제거 및 텍스트 정제 함수
def clean_text(text):
    # HTML 태그 제거 (더 엄격한 정규식 패턴 사용)
    text = re.sub(r'<[^>]+>', '', text)  # 모든 HTML 태그 제거
    text = re.sub(r'\s*</div>\s*$', '', text)  # 끝에 있는 </div> 태그 제거
    return text.strip()

# HS 코드 추출 패턴 정의 및 함수
# 더 유연한 HS 코드 추출 패턴
HS_PATTERN = re.compile(
    r'(?:HS\s*)?(\d{4}(?:[.-]?\d{2}(?:[.-]?\d{2}(?:[.-]?\d{2})?)?)?)',
    flags=re.IGNORECASE
)

def extract_hs_codes(text):
    """
    여러 HS 코드를 추출하고, 중복 제거 및 숫자만 남겨 표준화
    개선사항:
    - 단어 경계(\b) 제거로 더 유연한 매칭
    - 숫자만 있는 경우도 처리 가능
    - 최소 4자리 숫자 체크 추가
    """
    matches = HS_PATTERN.findall(text)
    hs_codes = []
    
    for raw in matches:
        # 숫자만 남기기
        code = re.sub(r'\D', '', raw)
        # 최소 4자리이고 중복이 아닌 경우만 추가
        if len(code) >= 4 and code not in hs_codes:
            hs_codes.append(code)
    
    # 만약 위 패턴으로 찾지 못하고, 입력이 4자리 이상의 숫자로만 구성된 경우
    if not hs_codes:
        # 순수 숫자만 있는 경우 체크
        numbers_only = re.findall(r'\d{4,}', text)
        for num in numbers_only:
            if num not in hs_codes:
                hs_codes.append(num)
    
    return hs_codes

def extract_and_store_text(json_file):
    """JSON 파일에서 head1과 text를 추출하여 변수에 저장"""
    try:
        # JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 데이터를 변수에 저장
        extracted_data = []
        for item in data:
            head1 = item.get('head1', '')
            text = item.get('text', '')
            if head1 or text:
                extracted_data.append(f"{head1}\n{text}")
        
        return extracted_data
    except Exception as e:
        print(f"오류 발생: {e}")
        return []

# 통칙 데이터 로드 (재사용을 위한 전역 변수)
general_explanation = extract_and_store_text('knowledge/통칙_grouped.json')

def lookup_hscode(hs_code, json_file):
    """HS 코드에 대한 해설 정보를 조회하는 함수"""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 각 설명 유형별 초기값 설정
        part_explanation = {"text": "해당 부에 대한 설명을 찾을 수 없습니다."}
        chapter_explanation = {"text": "해당 류에 대한 설명을 찾을 수 없습니다."}
        sub_explanation = {"text": "해당 호에 대한 설명을 찾을 수 없습니다."}

        # 1) 류(類) key: "제00류"
        chapter_key = f"제{int(hs_code[:2])}류"
        chapter_explanation = next((g for g in data if g.get('header2') == chapter_key), chapter_explanation)

        # 2) 호 key: "00.00"
        sub_key = f"{hs_code[:2]}.{hs_code[2:]}"
        sub_explanation = next((g for g in data if g.get('header2') == sub_key), sub_explanation)

        # 3) 부(部) key: "제00부"
        part_key = chapter_explanation.get('header1') if chapter_explanation else None
        part_explanation = next((g for g in data if (g.get('header1') == part_key)&(re.sub(r'제\s*(\d+)\s*부', r'제\1부', g.get('header1')) == part_key)), None)
        
        return part_explanation, chapter_explanation, sub_explanation
    
    except Exception as e:
        print(f"HS 코드 조회 오류: {e}")
        return ({"text": "오류가 발생했습니다."}, {"text": "오류가 발생했습니다."}, {"text": "오류가 발생했습니다."})

def get_hs_explanations(hs_codes):
    """여러 HS 코드에 대한 해설을 취합하는 함수 (마크다운 형식)"""
    all_explanations = ""
    for hs_code in hs_codes:
        explanation, type_explanation, number_explanation = lookup_hscode(hs_code, 'knowledge/grouped_11_end.json')

        if explanation and type_explanation and number_explanation:
            all_explanations += f"\n\n# HS 코드 {hs_code} 해설\n\n"
            all_explanations += f"## 📋 해설서 통칙\n\n"
            
            # 통칙 내용을 리스트 형태로 정리
            if general_explanation:
                for i, rule in enumerate(general_explanation[:5], 1):  # 처음 5개만 표시
                    all_explanations += f"### 통칙 {i}\n{rule}\n\n"
            
            all_explanations += f"## 📂 부(部) 해설\n\n{explanation['text']}\n\n"
            all_explanations += f"## 📚 류(類) 해설\n\n{type_explanation['text']}\n\n"
            all_explanations += f"## 📝 호(號) 해설\n\n{number_explanation['text']}\n\n"
            all_explanations += "---\n"  # 구분선 추가
    
    return all_explanations

# 질문 유형 분류 함수 (LLM 기반)
def classify_question(user_input):
    """
    LLM(Gemini)을 활용하여 사용자의 질문을 아래 네 가지 유형 중 하나로 분류합니다.
    - 'web_search': 물품 개요, 용도, 기술개발, 무역동향, 산업동향 등
    - 'hs_classification': HS 코드, 품목분류, 관세 등
    - 'hs_manual': HS 해설서 본문 심층 분석
    - 'overseas_hs': 해외(미국/EU) HS 분류 사례
    """
    system_prompt = """
아래는 HS 품목분류 전문가를 위한 질문 유형 분류 기준입니다.

질문 유형:
1. "web_search" : "뉴스", "최근", "동향", "해외", "산업, 기술, 무역동향" 등 일반 정보 탐색이 필요한 경우.
2. "hs_classification": HS 코드, 품목분류, 관세, 세율 등 HS 코드 관련 정보가 필요한 경우.
3. "hs_manual": HS 해설서 본문 심층 분석이 필요한 경우.
4. "overseas_hs": "미국", "해외", "외국", "US", "America", "EU", "유럽" 등 해외 HS 분류 사례가 필요한 경우.
5. "hs_manual_raw": HS 코드만 입력하여 해설서 원문을 보고 싶은 경우.

아래 사용자 질문을 읽고, 반드시 위 다섯 가지 중 하나의 유형만 한글이 아닌 소문자 영문으로 답변하세요.
질문: """ + user_input + """\n답변:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash", # 또는 최신 모델로 변경 가능
        contents=system_prompt,
        )
    answer = response.text.strip().lower()
    # 결과가 정확히 네 가지 중 하나인지 확인
    if answer in ["web_search", "hs_classification", "hs_manual", "overseas_hs", "hs_manual_raw"]:
        return answer
    # 예외 처리: 분류 실패 시 기본값
    return "hs_classification"

# 질문 유형별 처리 함수
def handle_web_search(user_input, context, hs_manager):
    # 웹검색 전용 컨텍스트로 수정
    web_context = """당신은 HS 품목분류 전문가입니다. 
사용자의 질문에 대해 최신 웹 정보를 검색하여 물품개요, 용도, 기술개발, 무역동향, 산업동향 등의 정보를 제공해주세요.
국내 HS 분류 사례가 아닌 일반적인 시장 정보와 동향을 중심으로 답변해주세요."""
    
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    
    prompt = f"{web_context}\n\n사용자: {user_input}\n"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config)
    
    return clean_text(response.text)

def handle_hs_classification_cases(user_input, context, hs_manager):
    """국내 HS 분류 사례 처리 (그룹별 Gemini + Head Agent)"""
    # 5개 그룹별로 각각 Gemini에 부분 답변 요청
    group_answers = []
    for i in range(5):  # 3 → 5로 변경
        relevant = hs_manager.get_domestic_context_group(user_input, i)
        prompt = f"{context}\n\n관련 데이터 (국내 관세청, 그룹{i+1}):\n{relevant}\n\n사용자: {user_input}\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        group_answers.append(clean_text(response.text))

    # Head Agent가 5개 부분 답변을 취합하여 최종 답변 생성
    head_prompt = f"{context}\n\n아래는 국내 HS 분류 사례 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"
    for idx, ans in enumerate(group_answers):
        head_prompt += f"[그룹{idx+1} 답변]\n{ans}\n\n"
    head_prompt += f"\n사용자: {user_input}\n"
    head_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=head_prompt
    )
    return clean_text(head_response.text)

def handle_hs_manual(user_input, context, hs_manager):
    # 예: HS 해설서 분석 전용 컨텍스트 추가
    manual_context = context + "\n(심층 해설서 분석 모드)"
    hs_codes = extract_hs_codes(user_input)
    explanations = get_hs_explanations(hs_codes) if hs_codes else ""
    prompt = f"{manual_context}\n\n관련 데이터:\n{explanations}\n\n사용자: {user_input}\n"
    # client.models.generate_content 사용
    response = client.models.generate_content(
        model="gemini-2.5-flash", # 모델명 단순화
        contents=prompt
    )
    return clean_text(response.text)

def handle_overseas_hs(user_input, context, hs_manager):
    """해외 HS 분류 사례 처리 (그룹별 Gemini + Head Agent)"""
    overseas_context = context + "\n(해외 HS 분류 사례 분석 모드)"
    
    # 5개 그룹별로 각각 Gemini에 부분 답변 요청
    group_answers = []
    for i in range(5):
        relevant = hs_manager.get_overseas_context_group(user_input, i)
        prompt = f"{overseas_context}\n\n관련 데이터 (해외 관세청, 그룹{i+1}):\n{relevant}\n\n사용자: {user_input}\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        group_answers.append(clean_text(response.text))

    # Head Agent가 5개 부분 답변을 취합하여 최종 답변 생성
    head_prompt = f"{overseas_context}\n\n아래는 해외 HS 분류 사례 데이터 5개 그룹별 분석 결과입니다. 각 그룹의 답변을 종합하여 최종 전문가 답변을 작성하세요.\n\n"
    for idx, ans in enumerate(group_answers):
        head_prompt += f"[그룹{idx+1} 답변]\n{ans}\n\n"
    head_prompt += f"\n사용자: {user_input}\n"
    head_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=head_prompt
    )
    return clean_text(head_response.text)