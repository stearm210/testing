#id定位就是通过元素的id进行定位，HTML规定id属性
#在整个HTML文档中是唯一的，前提是元素有id属性
'''
定位方法
element=driver.find_element_by_id(id)

例子
1#导入模块
from selenium import webdriver
2#实例化浏览器对象
driver=webdriver.Chrome()
3#打开网页
driver.get('http://www.baidu.com')

#实现对应需求
使用id定位，输入用户名:admin
username=driver.find_element_by_id('userA')
之后再将得到的定位输入。输入方法：元素对象.send_keys('内容‘)
username.send_keys('admin')

使用id定位,输入密码:123456
password=driver.find_element_by_if('password')
password.send_keys('123456)

3秒后关闭浏览器窗口

4#展示效果
time.sleep(3)
5#关闭页面
driver.quit()

'''

#当存在id值时，优先使用id值解决问题










