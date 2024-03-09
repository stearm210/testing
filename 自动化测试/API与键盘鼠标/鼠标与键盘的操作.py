
#在selenium中将操作鼠标的方法封装在ActionChains类中
#实例化对象

from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

#打开注册页面A，在用户名文本框上点击鼠标右键
#0.定位目标元素
username=driver.find_element_by_id('userA')
#1.实例化鼠标对象（关联浏览器对象）
action=ActionChains(driver)

#2.调用鼠标方法
#鼠标右键只能展示右键菜单内容，菜单中的元素无法操作
action.context_click(username)

#3.执行方法:该方法必须调用(使用.perform（）调用)，否则上述代码无效
action.perform()

#展示效果
sleep(3)
driver.quit()











































































