import pytest
#函数形式
'''
函数形式要求函数名以test开头
'''
def test_func():
    "测试函数"
    print('测试块')

#使用主函数写法
if __name__ == '__main__':
    pytest.main(['-s','test3.py'])





























