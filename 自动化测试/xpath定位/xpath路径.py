'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')
#对应需求
1.使用绝对路径定位用户名输入框，输入:admin
driver.find_element_by_xpath('/html/body/div[1]/div/main/div[1]/div[1]/div[2]/div[2]/main/section/input').send_keys('admin')
2.暂停2秒
sleep(2)

3.使用相对路径定位密码输入框，输入：123
driver.find_element_by_xpath('').send_keys('123')

#展示效果
sleep(3)
#退出浏览器
driver.quit()

'''

























































