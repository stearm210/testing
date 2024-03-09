'''
from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')


#打开页面，将红色方框拖拽到蓝色方框上
red=driver.find_element_by_id('divl')
blue=driver.find_element_by_id('div2')

1.实例化鼠标对象（关联浏览器对象）
action=ActionChains(drievr)
2.调用方法（传入目标元素）
action.drag_and_drop(red,blue)
3.执行方法
action.perform()

sleep(3)
driver.quit()
'''






































