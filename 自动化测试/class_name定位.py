'''
class_name定位是根据元素class属性来定位元素
HTML通过使用class来定义元素样式。
#前提：元素有class属性
#如果class有多个属性值，只能使用其中一个


1#导入模块
from selenium import webdriver
2#实例化浏览器对象
driver=webdriver.Chrome()
3#打开网页
driver.get('http://www.baidu.com')

#实现需求
1.通过class_name定位电话号码A，输入：18611111111111
tel=driver.find_element_by_class_name('telA')
tel.send_keys('123333333333')

2.通过class_name定位电子邮件A，输入：123@qq.com
mail=driver.find_element_by_class_name('emailA dzyxA')
mail.send_keys('123@qq.com')

3.3秒之后关闭浏览器窗口
time.sleep(3)

#方法名是class_name，但是要找元素的class属性值
#如果元素的class属性值存在多个值，只能使用其中任意一个


'''




























