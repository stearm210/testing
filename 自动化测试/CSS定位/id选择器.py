
'''
说明：根据元素id属性来选择
格式：#id
例子：#userA<选择id属性值为userA的元素>

'''

'''
from time import sleep
from selenium import webdriver
#实例化对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#需求：打开注册A.html页面
1.使用CSS id选择器定位,输入admin
driver.find_element_by_css_selector('#userA').send_keys('admin)
2.属性选择器

3.class选择器

4.元素选择器


#展示效果
sleep(3)
#退出浏览器
driver.quit()




'''

















