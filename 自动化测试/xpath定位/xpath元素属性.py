'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#利用元素属性策略
#打开页面，右键检查，选择copy->xpath复制下来就是: //标签名[@属性名='属性值']



语法1：//标签名[@属性名='属性值']
语法2://*[@属性名='属性值']
#利用元素属性定位用户输入框，输入:admin
driver.find_element_by_xpath('input[@placeholder="输入用户名"').send_keys('admin')
driver.find_element_by_xpath('//*[@placeholder="输入用户名"]').send_keys('admin)
#使用xpath策略，需要在浏览器工具中根据策略语法，组装策略值，验证之后再放入代码中

#如果出现多个class值时，可以全部复制完
diver.find_element_by.xpath('//*[@class="emailA dzyxA"]').send_keys('123@qq.com')


#展示效果
sleep(3)

#退出浏览器
driver.quit()


'''




























