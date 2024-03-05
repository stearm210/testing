'''
用户名：#username
密码: #password
验证码:#verify_code
登录:.J-login-submit
'''
import time

#导包
from selenium import webdriver
from selenium.webdriver.common.by import By

#创建浏览器
driver=webdriver.Chrome()#谷歌浏览器窗口
driver.maximize_window()#浏览器最大窗口
#访问页面
driver.get("https://login.taobao.com/member/login.jhtml?redirectURL=http%3A%2F%2Fi.taobao.com%2Fmy_taobao.htm%3Fspm%3Da21bo.jianhua.1997525045.1.344b2a89FzrR9G")#访问想要测试的页面，如淘宝
#页面操作
#定位用户名
driver.find_element(By.CSS_SELECTOR,"#username").send_keys("12333333")
#定位密码
driver.find_element(By.CSS_SELECTOR,"#password").send_keys("12333333")
driver.find_element(By.CSS_SELECTOR,"#verify_code").send_keys("12333333")
#页面静止
time.sleep(2)
driver.find_element(By.CSS_SELECTOR,"#.J-login-submit").send_keys("12333333")
#退出浏览器
time.sleep(3)
driver.quit()







































