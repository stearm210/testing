'''
from time import sleep
from selenium import webdriver
from selenium.webdriver import  ActionChains
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

#需求：打开注册A页面，完成一下操作
1.输入用户名：admin，暂停2秒，删除1
username=driver.find_element_by_id('userA')
username.send_keys('admin1')
sleep(2)
#删除
username.send_keys(Keys.BACK_SPACE)


2.全选用户名：admin，暂停2秒
username.send_keys(Keys.CONTROL,'a')#windows系统
username.send_keys(Keys.COMMAND,'a')#macos系统
sleep(2)


3.复制用户名：admin，暂停2秒
username.send_keys(Keys.CONTROL,'c')#windows系统
username.send_keys(Keys.COMMAND,'c')#macos系统
sleep(2)




4.粘粘到密码框




#展示效果
sleep(3)
#退出浏览器
driver.quit()

'''






























