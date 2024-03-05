'''
import unittest

from 软件测试python练习.test1 import TestDemo1

#使用套件对象添加用例方法
suite=unittest.TestSuite()#实例化套件对象
#套件对象.测试类名
suite.addTest(unittest.makeSuite(TestDemo1))


#实例化运行对象
runner=unittest.TextTestRunner()

#使用运行对象去执行套件对象
runner.run(suite)

'''

































































