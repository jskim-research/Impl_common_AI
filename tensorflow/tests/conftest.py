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
    - 기본 실행 명령어는 Fail => F, pass => . 으로만 표시하는데 여기선 각 테스트 함수 실행 결과 출력 (아마 fail 시에만)
- vv 옵션
    - 더 자세한 실행 결과 설명
- s 옵션
    - Failed 건에 대한 stdout, stderr 메세지 캡쳐 기능 비활성화 (== --capture=no)
- r 옵션
    - short test summary info에 나올 정보 지정 (e.g., 디폴트 옵션 -rfE는 [f]ailed 와 [E]rror 건 출력
- --capture
    - Passed 건에 대해 stdout, stderr 메세지 캡쳐하고 싶으면 --capture=tee-sys 사용
- --disable-pytest-warnings // deprecate 등의 warning 출력 생략, 그러나 warning이 있다는 것은 계속 알려줌

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
