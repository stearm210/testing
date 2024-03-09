from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

wd = webdriver.Chrome()
wd.get('http://www.baidu.com')
# wd是webdriver对象，10是最长等待时间，0.5是每0.5秒去查询对应的元素。until后面跟的等待具体条件，EC是判断条件，检查元素是否存在于页面的 DOM 上。
login_btn = WebDriverWait(wd, 10, 0.5).until(EC.presence_of_element_located((By.ID, "s-top-loginbtn")))
# 点击元素
login_btn.click()











