# 导入强制等待模块
import time
from selenium import webdriver

wd = webdriver.Chrome()
wd.get('https://www.baidu.com')
# 强制等待5秒
time.sleep(5)




























