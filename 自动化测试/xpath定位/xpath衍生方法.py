from time import sleep
from selenium import webdriver

#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')
#定位文本信息
#//*[text()="文本信息"]:
# 通过文本信息定位目标元素（要求全部文本内容）
driver.find_element_by_xpath('//*[text()="访问"]').click()

#//*[contains(@属性名,'属性值的部分内容')]
#通过给定属性值的部分进行元素定位



#展示效果
sleep(3)
#退出浏览器
driver.quit()


























