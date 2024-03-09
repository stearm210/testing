from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')


#打开注册页面A，输入用户名，暂停3秒之后，双击鼠标左键，选中admin
username=driver.find_element_by_id('userA')
username.send_keys('admin')

'''
1.实例化
action=ActionChains(driver)
2.调用
action.double_click(username)
3.执行
action.perform()


'''

#展示效果
sleep(3)
driver.quit()







































