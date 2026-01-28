"""
API 재시도 로직 유틸리티
- Gemini API 503/429 에러에 대한 지수 백오프 재시도
- 429 에러 시 API RetryInfo 파싱하여 정확한 대기 시간 사용
- Streamlit UI 재시도 상태 표시
"""

import time
import functools
import re
import random
from typing import Callable, Any, Optional
from google.genai.errors import APIError


def extract_retry_delay_from_error(error: APIError) -> Optional[float]:
    """
    429 에러 메시지에서 RetryInfo의 retryDelay 추출

    Args:
        error: APIError 객체

    Returns:
        추출된 대기 시간(초), 없으면 None

    Example:
        "Please retry in 2.273621767s." -> 2.273621767
    """
    if not error or error.code != 429:
        return None

    try:
        # 메시지에서 "Please retry in X.Xs" 패턴 찾기
        message = str(error.message) if error.message else ""
        match = re.search(r'retry in ([0-9.]+)s', message, re.IGNORECASE)

        if match:
            delay = float(match.group(1))
            # 안전을 위해 최소 0.5초, 최대 10초로 제한
            return max(0.5, min(delay, 10.0))

    except (ValueError, AttributeError):
        pass

    return None


def retry_on_api_error(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    retryable_codes: tuple = (503, 429),
    ui_callback: Optional[Callable[[int, float, str], None]] = None
):
    """
    Gemini API 호출 재시도 데코레이터

    Args:
        max_retries: 최대 재시도 횟수 (기본 3회)
        initial_delay: 초기 대기 시간 (기본 0.5초)
        backoff_factor: 백오프 증가 배수 (기본 2.0)
        retryable_codes: 재시도 가능한 HTTP 코드 (기본 503, 429)
        ui_callback: UI 업데이트 콜백 함수 (재시도 횟수, 대기 시간, 메시지)

    Returns:
        데코레이터 함수

    Example:
        @retry_on_api_error(max_retries=3)
        def call_gemini_api():
            return client.models.generate_content(...)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except APIError as e:
                    last_exception = e

                    # 재시도 가능한 에러인지 확인
                    if e.code not in retryable_codes:
                        raise

                    # 마지막 시도면 더 이상 재시도하지 않음
                    if attempt >= max_retries - 1:
                        raise

                    # 대기 시간 계산
                    if e.code == 429:
                        # 429 에러: API가 알려준 정확한 시간 사용
                        api_delay = extract_retry_delay_from_error(e)
                        if api_delay:
                            wait_time = api_delay
                        else:
                            # RetryInfo 없으면 안전하게 3초 대기
                            wait_time = 3.0

                        # 병렬 재시도 충돌 방지: 랜덤 지터 추가 (0.2~0.8초)
                        jitter = random.uniform(0.2, 0.8)
                        wait_time += jitter
                    else:
                        # 503 에러: 기존 지수 백오프 사용
                        wait_time = initial_delay * (backoff_factor ** attempt)

                    # UI 콜백 호출 (있는 경우)
                    if ui_callback:
                        error_msg = f"API 오류 {e.code}: {e.message}"
                        ui_callback(attempt + 1, wait_time, error_msg)

                    # 대기
                    time.sleep(wait_time)

                except Exception as e:
                    # 예상치 못한 에러는 즉시 발생
                    raise

            # 모든 재시도 실패 시 마지막 예외 발생
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def create_retry_callback_for_streamlit(container=None):
    """
    Streamlit UI를 위한 재시도 콜백 함수 생성

    Args:
        container: Streamlit 컨테이너 (optional)

    Returns:
        콜백 함수
    """
    def callback(attempt: int, wait_time: float, error_msg: str):
        import streamlit as st

        target = container if container else st
        target.warning(
            f"재시도 {attempt}/3: {error_msg}\n\n"
            f"{wait_time:.1f}초 후 다시 시도합니다..."
        )

    return callback


def retry_api_call(
    func: Callable,
    max_retries: int = 3,
    ui_container=None
) -> Any:
    """
    함수를 직접 재시도하는 헬퍼 함수 (데코레이터 대신 사용)

    Args:
        func: 호출할 함수
        max_retries: 최대 재시도 횟수
        ui_container: Streamlit 컨테이너 (optional)

    Returns:
        함수 실행 결과

    Example:
        result = retry_api_call(
            lambda: client.models.generate_content(...),
            max_retries=3,
            ui_container=st.container()
        )
    """
    ui_callback = None
    if ui_container:
        ui_callback = create_retry_callback_for_streamlit(ui_container)

    @retry_on_api_error(max_retries=max_retries, ui_callback=ui_callback)
    def _wrapped():
        return func()

    return _wrapped()


# 간단한 테스트
if __name__ == '__main__':
    print("API 재시도 유틸리티 테스트")

    # 테스트용 함수 (처음 2번 실패, 3번째 성공)
    call_count = 0

    @retry_on_api_error(max_retries=3)
    def test_api_call():
        global call_count
        call_count += 1
        print(f"API 호출 시도 #{call_count}")

        if call_count < 3:
            # 503 에러 시뮬레이션
            class MockAPIError(APIError):
                def __init__(self):
                    self.code = 503
                    self.message = "The model is overloaded"
                    self.status = "UNAVAILABLE"
            raise MockAPIError()

        return "성공!"

    try:
        result = test_api_call()
        print(f"결과: {result}")
    except Exception as e:
        print(f"최종 실패: {e}")
