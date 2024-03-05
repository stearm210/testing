#导入模块
import time

from selenium import webdriver

#实例化浏览器对象:类名()
driver=webdriver.Chrome()

#打开网页:必须包含协议头
driver.get('http://www.baidu.com')
#观察效果
time.sleep(3)
#关闭页面
driver.quit()






























































