'''
案例：假设现在有一个页面，页面中有两个窗口，两个窗口分别有
注册操作，两个窗口分别叫做A、B，分别对两个窗口填写注册信息

'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#1.填写主页面注册信息
driver.find_element_by_id('user').send_keys('admin')
driver.find_element_by_id('password').send_keys('123456')

#2.填写注册页面A中的注册信息
#由于这里是注册A中的信息，因此如果目标元素存在frame中
#需要先执行切换对应的frame，之后定位元素。
#如果找不到，就用iframe中的一个唯一的特征值进行操作
#这里选择能代表frame元素唯一性的特征值
driver.switch_to.frame('idframe1')
driver.find_element_by_id('userA').send_keys('admin1')
driver.find_element_by_id('passwoed').send_keys('123456')

#3.如果已经在A页面中了，这个时候需要退出A页面回到主页面之后再进入B页面
#切换回默认页面：如果连续切换多个frame，必须先回到默认页面
#只有上面那样，才可以切换到下一个frame
driver.switch_to.default_content()
#切换frame
driver.switch_to.frame('myframe2')

#填写注册B中的注册信息
driver.find_element_by_id('userB').send_keys('admin2')
driver.find_element_by_id('passwordB').send_keys('123456')

#展示效果
sleep(3)
#退出浏览器
driver.quit()








































































































































































































