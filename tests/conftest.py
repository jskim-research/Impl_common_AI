"""pytest configuration

tests 폴더 내의 모든 test code 실행 방법\n
- root directory 이동
- python -m pytest tests (tests 폴더 내 test_*.py 또는 *_test.py 파일 모두 실행)

일부 test code 실행 방법\n
- python -m pytest {디렉토리명}/{테스트파일명}.py

pytest option (https://velog.io/@sangyeon217/pytest-execution)
- k 옵션
    - pytest {테스트파일명}.py -k {테스트함수명} // 특정 테스트 함수만 실행
- v 옵션
    - 기본 실행 명령어는 Fail => F, pass => . 으로만 표시하는데 여기선 각 테스트 함수 실행 결과 출력
- vv 옵션
- s 옵션
- r 옵션

중복 class 공유 방법

- conftest.py
from src.xx import Class

@pytest.fixture \n
def class():
    class = Class() \n
    return class \n

- test_model.py
def test_func1(class):
    '''Test functionality of func1'''
    asset class.func1(...) == some_value
"""
import pytest
