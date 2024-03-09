#解决元素之间相同属性重名的问题
#格式如下
#//*[@name='tel' and @class='tel']


'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#属性和逻辑结合：解决目标元素单个属性和属性无法定位为一个元素的问题时使用
#语法：//*[@属性1="属性值1" and @属性2="属性值2"]
#多个属性值由and链接，每一个属性由@开头，可以根据需求使用更多属性值
driver.find_element_by_xpath('//*[@name="user" and @class="login"]').send_keys('admin')





'''












































