'''
from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')


#定位目标元素
btn=driver.find_element_by_tag_name('button')

#实例化鼠标对象
action=ActionChains(driver)

#调用鼠标方法
#方法执行时，不动鼠标
#鼠标实例化之后，使用点方法调用鼠标方法
action.move_to_element(btn)

#执行
action.perform()

#展示效果
sleep(3)
#退出浏览器
driver.quit()

'''










































