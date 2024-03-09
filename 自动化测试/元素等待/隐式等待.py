'''

from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

#设置隐式等待
driver.implicitly_wait(10)

#定位元素并输入
driver.find_element_by_id('userA').send_keys('admin')


#展示效果
sleep(3)

#退出浏览器
driver.quit()

'''





































