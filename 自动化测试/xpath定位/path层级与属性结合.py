#如果通过元素自身的信息不方便直接定位到该元素，
# 则可以先定位到其父级元素，然后再找到该元素格式
#格式为：//*[@id='p1']/input


'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#层级与属性结合：目标元素无法直接定位，可以考虑先定位
#其父层级或者祖辈层级，再获取目标元素
driver.find_element_by_xpath('*[@id="p1"]/input').send_keys('admin')




#展示效果
sleep(3)
#退出浏览器
driver.quit()






'''





















