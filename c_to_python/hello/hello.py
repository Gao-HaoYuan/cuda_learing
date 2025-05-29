# my_project/hello.py
from . import hello_module

def greet():
    print(hello_module.hello())  # 调用 C 扩展中的 hello 函数

def add_two_numbers(a, b):
    return hello_module.add(a, b)  # 调用 C 扩展中的 add 函数
