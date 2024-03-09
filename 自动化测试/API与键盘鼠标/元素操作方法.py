'''
1.

'''

from time import sleep
from selenium import webdriver
#实例化对象
driver=webdriver.Chrome()
driver.get('')

#需求
'''
1.脚本输入用户名:admin 密码:123456 电话:1866666666 电子邮件
driver.find_element_by_id('username').send_keys('admin')
driver.find_element_by_id('password').send_keys('123456')
tel=driver.find_element_by_id('telA')
tel.send_keys('1866666666')
driver.find_element_by_id('email').send_keys('123@qq.com')

2.间隔3秒，修改电话号码：18666
在使用操作时，一般对于输入框元素，先清空，再进行输入
这样会避免操作错误

sleep(3)
tel.clear()
tel.send_keys('1866666666')

3.间隔3秒：点击“注册”按钮
sleep(3)
driver.find_element_by_css_selector('button').click()


4.间隔3秒：关闭浏览器

5.元素定位方法不限

'''

sleep(3)
driver.quit()









