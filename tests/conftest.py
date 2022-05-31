"""pytest configuration

tests 폴더 내의 모든 test code 실행 방법

- root directory 이동
- python3 -m tests 입력

중복 class 공유 방법

- conftest.py
from src.xx import Class

@pytest.fixture \n
def class():
    class = Class() \n
    return class \n

- test_code.py
def test_func1(class):
    '''Test functionality of func1'''
    asset class.func1(...) == some_value
"""
import pytest