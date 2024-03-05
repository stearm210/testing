# 列表
var1 = 'aaa'
var2 = 'adasidjiasjd'
print("var1[0]:", var1[0])
print("已经更新字符:", var1[:2] + 'ooo')
list1 = ['Google', 'Runoob', 1997, 2000]
list2 = [1, 2, 3, 4, 5]
list3 = ["a", "b", "c", "d"]
list4 = ['red', 'green', 'blue', 'yellow', 'white', 'black']

print(list1[0])
print(list2[1])
print(list3[2])

nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print(nums[0:4])
nums[2] = 109
print(nums[0:4])
list1.append('op')
print(list1)

list = ['Google', 'Runoob', 1997, 2000]
print("原始列表 : ", list)
del list[2]
print("删除第三个元素 : ", list)

import operator

a = [1, 2]
b = [2, 3]
c = [2, 3]
print("operator.eq(a,b): ", operator.eq(a, b))
print("operator.eq(c,b): ", operator.eq(c, b))
print(len(list))
print(max(nums))

# 元组
tup1 = ('Google', 'Runoob', 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7)
tup3 = "a", "b", "c", "d"  # 不需要括号也可以
print(type(tup3))
print("tup1[0]: ", tup1[0])
print("tup2[1:5]: ", tup2[1:5])
print(len(tup2))

tup = ('Google', 'Runoob', 'Taobao', 'Wiki', 'Weibo', 'Weixin')
print(tup[1])
print(tup[-2])
print(tup[1:4])

# 字典
# 使用大括号 {} 来创建空字典
emptyDict = {}
print(emptyDict)
print("Length:", len(emptyDict))
print(type(emptyDict))  # 类型

tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
# 访问值
print("tinydict['Name']: ", tinydict['Name'])
print("tinydict['Age']: ", tinydict['Age'])
# 修改字典
tinydict['Age'] = 8
tinydict['School'] = "菜鸟教程"
print(tinydict['Age'])
print(tinydict['School'])
# 注意，健的选择可以是数字、字符串或者元组，但是无法使用列表

tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
print(len(tinydict))
# 输出字典，可以打印的字符串表示
print(str(tinydict))

# 集合，无序不重复元素序列
set1 = {1, 2, 3, 4}
set2 = set([4, 5, 6, 7])
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)
# 判断元素是否在集合内部
print('orange' in basket)
print('crabgrass' in basket)
thisset = set(("Google", "Runoob", "Taobao"))
thisset.add("Facebook")
print(thisset)

thisset = set(("Google", "Runoob", "Taobao"))
print(len(thisset))

# 条件语句循环
# 条件控制
var1 = 100
if var1:
    print("1 - if 表达式条件为 true")
    print(var1)
var2 = 0
if var2:
    print("2 - if 表达式条件为 true")
    print(var2)
print("Good bye!")

'''
age = int(input("请输入你家狗狗的年龄: "))
print("")
if age <= 0:
    print("你是在逗我吧!")
elif age == 1:
    print("相当于 14 岁的人。")
elif age == 2:
    print("相当于 22 岁的人。")
elif age > 2:
    human = 22 + (age - 2) * 5
    print("对应人类年龄: ", human)
'''
### 退出提示
# input("点击 enter 键退出")
'''
num = int(input("输入一个数字："))
if num % 2 == 0:
    if num % 3 == 0:
        print("你输入的数字可以整除 2 和 3")
    else:
        print("你输入的数字可以整除 2，但不能整除 3")
else:
    if num % 3 == 0:
        print("你输入的数字可以整除 3，但不能整除 2")
    else:
        print("你输入的数字不能整除 2 和 3")
'''

# 循环语句
num = 100
sum = 0
count = 1
while count <= num:
    sum = sum + count
    count += 1
print("1到100和", sum)

sites = ["Baidu", "Google", "Runoob", "Taobao"]
for site in sites:
    print(site)

sites = ["Baidu", "Google", "Runoob", "Taobao"]
for site in sites:
    if site == "Runoob":
        print("菜鸟教程!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")

for i in range(5):
    print(i)
for i in range(5, 9):
    print(i)
for i in range(0, 10, 3):
    print(i)

# break与continue的区别：break为跳出循环，continue为跳过剩下语句
# 进行下一轮循环
n = 5
while n > 0:
    n -= 1
    if n == 2:
        break;
    print(n)
print("循环结束")

for letter in 'Runoob':  # 第一个实例
    if letter == 'b':
        break
    print('当前字母为 :', letter)
var = 10  # 第二个实例
while var > 0:
    print('当前变量值为 :', var)
    var = var - 1
    if var == 5:
        break

print("Good bye!")

my_list = ['google', 'runoob', 'taobao']
print(my_list[0])
print(my_list[1])
print(my_list[2])

a, b = 0, 1
while b < 1000:
    print(b, end=',')
    a, b = b, a + b
print("\n")


# 面向对象
class myclass:
    i = 111
    factor1 = ' '
    factor2 = ' '

    def __init__(self, a, b):
        self.factor1 = a
        self.factor2 = b

    def f(self):
        return 'nihao'


x = myclass(1, 2)
print(x.factor1, x.factor2)


class people:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0

    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("%s 说: 我 %d 岁。" % (self.name, self.age))


# 实例化类
p = people('runoob', 10, 30)
p.speak()


# 继承父类,同时多态
class student(people):
    grade = ' '

    # 继承之后重新添加参数
    def __init__(self, n, a, w, g):
        # 调用父类构造函数
        people.__init__(self, n, a, w)
        self.grade = g

    # 覆写父类方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))


s = student('ken', 10, 60, 3)
s.speak()


# 多继承
# 另一个类，多继承之前的准备
class speaker():
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s" % (self.name, self.topic))


# sample继承两个类
class sample(speaker, student):
    a = ''

    def __init__(self, n, a, w, g, t):
        student.__init__(self, n, a, w, g)
        speaker.__init__(self, n, t)


test = sample("Tim", 25, 80, 4, "Python")
test.speak()  # 方法名同，默认调用的是在括号中参数位置排前父类的方法


# 方法重写
class Parent:  # 定义父类
    def myMethod(self):
        print('调用父类方法')


class Child(Parent):  # 定义子类
    def myMethod(self):
        print('调用子类方法')


c = Child()  # 子类实例
c.myMethod()  # 子类调用重写方法
super(Child, c).myMethod()  # 用子类对象调用父类已被覆盖的方法


# 私有类方法
class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0  # 公开变量

    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)  # 类内部可以调用私有属性


counter = JustCounter()  # 用一个类
counter.count()
counter.count()
print(counter.publicCount)


# print(counter.__secretCount)  # 报错，实例不能访问私有变量


# 私有方法与公有方法混用,外部无法调用私有方法
class Site:
    def __init__(self, name, url):
        self.name = name  # public
        self.__url = url  # private

    def who(self):
        print('name  : ', self.name)
        print('url : ', self.__url)

    def __foo(self):  # 私有方法
        print('这是私有方法')

    def foo(self):  # 公共方法
        print('这是公共方法')
        self.__foo()


x = Site('菜鸟教程', 'www.runoob.com')
x.who()  # 正常输出
x.foo()  # 正常输出
# x.__foo()  # 报错


# 文件打开方式
'''
文件的打开方式.read()
open('a.txt','w',encoding='utf-8')
f.write('好好学习')写入文件
f.close()关闭文件
'''
f = open('a.txt', 'w', encoding='utf-8')
f.write('好好学习')
f.close()

f = open('a.txt', 'r', encoding='utf-8')
buf = f.read()
print(buf)
f.close()

# with open()打开文件,不用写关闭代码，会自动关闭
with open('a.txt', 'a', encoding='utf-8') as f:
    f.write('aooooo')

# 按行读取文件内容
with open('b.txt', encoding='utf-8') as f:
    buf = f.readline()
    print(buf)
    print(f.readline())

with open('b.txt', encoding='utf-8') as f:
    while True:
        # 按行读取，行数为0时会读取完毕
        buf = f.readline()  # 读取行数
        if len(buf) == 0:
            break
        else:
            print(buf, end='')

# json文件处理问题
'''
json文件后缀是.json
json中主要的数据类型是对象和数组
一个json文件时一个对象或者数组,要不是{}，要不是数组[]
每个数据之间有逗号隔开，最后一个数据后面不需要写逗号
json中的字符串必须使用双引号
json文件中的数据类型
数字类型：----->int float
string类型：--->str
布尔类型true,false--->True,False
null--->None

.load为读入文件操作
.get为获得数据操作
'''
'''
#读取json文件
import json
with open('info.json',encoding='utf-8') as f:
    #读入json文件操作
    result=json.load(f)
    print(type(result))
    #获得姓名
    print(result.get('name'))
    #获得年龄
    print(result.get('age'))
    #获得城市
    print(result.get('address').get('city'))
    pass
'''
'''
json实践用例
with open('cao.json',encoding='utf-8') as f:
    data=json.load(f)
    print(data)
    new_list=[]
    for i in data:
        new_list.append((i.get('username'),i.get('password'),i.get('expect')))
    print(new_list)
    
json读入操作
my_list=[('admin', '123456', '登录成功'), ('root', '123456', '登录失败')]
with open('cao.json','w',encoding='utf-8') as f:
    json.dump(my_list,f,ensure_ascii=False,indent=4)
'''

# 异常问题，捕获异常问题
# 可能输入的东西与程序中想要的东西不一致，因此产生异常
try:
    num = input('输入一个数字')
    num = int(num)
    print(num)
except ValueError:#报错类型
    print("输入正确的数字")
except ZeroDivisionError:
    print("除数不能为0")
# 异常捕获问题：
'''
try:
    书写可能发生异常的代码
except:
     发生了异常执行的代码
else:
     没有发生异常时执行的代码
finally:
    不管有没有发生异常，都会执行的代码
'''

'''
#模块的导入以及运用
import random
import json
random.randint(a,b)
json.load()
json.dump()
'''
'''
from 模块名 import 工具名
工具名
from random import randint
from json import load,dump
randint(a,b)
load()
dump()
'''


#unittest框架（断言判断）
'''
framework
unittest自动化测试方面
使用pytest
能够组织多个用例的执行
提供丰富的断言方法
生成测试报告

TestCase核心模块，TestCase(测试用例)，在这个文件中书写
用例代码。
TestSuit测试套件，用于管理组装多个TestCase
TestRunner用于执行TestSuite测试套件
TestLoader测试加载，对TestSuit测试套件功能上的补充，管理组装多个TestCase
Fixture(测试夹具)，书写在TestCase中，可以在每个方法执行前后都会执行的内容

学习testcase的书写
'''
'''
import unittest
#TestDemo用于继承unittest模块中的TestCase类
class TestDemo1(unittest.TestCase):
    def test_method1(self):
        print('测试方法1')
    def test_method2(self):
        print('测试方法2')

'''
'''
import unittest
from tools import add
class TestAdd(unittest.TestCase):
    def test_method1(self):
        if add(1,2)==3:
            print('测试通过')
        else:
            print('测试不通过')
'''


#testloader(测试加载)
'''
testloader(测试加载)，用于组装测试用例
'''
#导包
import  unittest
#实例化加载对象并添加用例
#unittest.TestLoader().discover('用例所在路劲‘，’用例所在代码文件名‘)
#路径使用相对路劲，文件名可以使用*表示多个字符
suite=unittest.TestLoader().discover('./case','test*.py')

#实例化运行对象
runner=unittest.TextTestRunner()
#执行
runner.run(suite)



#fixture测试夹具
#fixture测试夹具是一种代码结构
#这个代码结构有：方法级别、类级别和模块级别
#某些场景下会自动执行
#1.方法级别
'''
#在方法执行之前
def setup(self):
    在每个测试方法执行之前执行
    pass
#在方法执行之后
def teardown(self):
    在每个测试方法执行之后执行
    pass
'''
#2.类级别
'''
在每个测试类中所有方法执行前后，都会调用的结构(在整个类中，执行之前之后各一次)
类级别Fixture方法，是一个类方法
类中所有方法之前
@classmethod
def setupClass(cls):
    pass

#类中所有方法之后
@classmethod
def teardownClass(cls):
    pass
'''
#3.模块级别
'''
在每个代码文件执行前后执行的代码结构
#代码文件之前
def setupModle():
    pass
#代码文件之后
def teardownModule():
    pass

方法级别和类级别的前后方法，不需要同时出现，根据用例代码的需要
自行选择使用
'''

#案例
'''
打开浏览器（整个测试过程中只打开一次浏览器   类级别
输入网址（每个测试方法都需要一次          方法级别
输入用户数据（不同测试数据）             测试方法
关闭当前页面（每个测试方法需要一次        方法级别
关闭浏览器（整个测试过程中只关闭一次浏览器   类级别

'''
import  unittest
class TestLogin(unittest.TestCase):
    #方法阶段(方法才会执行这个)
    def setUp(self):
        "每个测试方法执行之前都会先调用的方法"
        print('输入网址.....')
    def tearDown(self) -> None:
        "每个测试方法执行之后都会调用的方法"
        print('关闭当前页面......')

    #类级别(方法执行完之后就会执行这个)
    @classmethod
    def setUpClass(cls) -> None:
        print('1.打开浏览器')
    @classmethod
    def tearDownClass(cls) -> None:
        print('5.关闭浏览器')

    def test_1(self):
        print('输入正确的用户名密码验证码,登录1')
    def test_2(self):
        print('输入错误的用户名密码验证码,登录2')


#断言
'''
让程序代替人工自动的判断预期结果和实际结果是否相符
断言结果两个，True，False。true是用例通过
false是代码抛出异常，用例不通过
#注意，在unittest中使用断言是，需要通过self.断言方法实验


self.assertEqual(预期结果，实际结果)#判断预期结果与实际结果是否相等
1.如果相等，用例通过
2.如果不相等，用例不通过，抛出异常

assertln
self.assertln(预期结果，实际结果)#判断预期结果是否包含在实际结果中
1.包含，用例通过
#以下这两种都是包含的
asserIn('admin','admin')
asserIn('admin','adminnnnnnn')
2.不包含，用例不通过，有异常
#下面这种方法不包含
assertIn('admin','adddddmin')

'''

#参数化
'''
参数化，在测试方法中，使用变量来代替具体的测试数据，
然后使用传参的方法将测试数据传递给方法的变量,可以对
重复的代码自动化使用

工作中的场景：
1.测试数据一般放在json文件中
2.使用代码读取json文件，提取我们想要的数据
使用参数化，必须需要安装插件来完成
'''
'''
#导入包使用parameterized
import unittest
from parameterized import parameterized
#组织测试数据
data=[
    ('admin','123456','登录成功')
    ('root','123456','登录失败')
    ('admin','123456','登录失败')
]
#书写测试方法
class TestLogin(unittest.TestCase):
    @parameterized.expand(data)
    def test_login(self,username,password,expect):
        self.assertEquals(expect, login(username,password))
'''

























































































































