#测试类方式
#使用测试类方法时，需要将test中的t进行大写
import pytest


class TestDemo(object):
    "测试实例类"

    #对于测试方法，则需要以test开头
    def test_method1(self):
        "测试方法1"
        print('测试方法1')
    def test_method2(self):
        "测试方法2"
        print('测试方法2')

#使用主函数写法
'''
主函数语法：
pytest.main(['-s','文件名.py'])
'''
if __name__ == '__main__':
    pytest.main(['-s','test2.py'])



























