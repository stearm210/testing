'''
import unittest
from test1 import TestAdd
#实例化测试套件
suite=unittest.TestSuite()
#添加测试方法
suite.addTest(unittest.makeSuite(TestAdd))

#实例化测试执行对象
runner=unittest.TextTestRunner()
runner.run(suite)
'''


















