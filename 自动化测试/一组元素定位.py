#导入模块
import time

from selenium import webdriver

#实例化浏览器对象:类名()
driver=webdriver.Chrome()

#打开页面
driver.get('www.baidu.com')
#通过name属性值定位用户名和密码完成操作
#假设name属性是一样的，可以使用列表的方法，通过下标进行定位
driver.find_element_by_name('AAA').send_keys('admin')
elements=driver.find_elements_by_name('AAA')
print(elements)
#元素定位方法如果带有s(如elements)，则执行结果返回的是列表数据类型，里面的数据是多个元素对象
#可以通过列表的下标获取对应目标的元素对象，再执行操作
elements[1].send_keys('123456')


#当标签名中有input时，也可以使用这种方法
#使用列表进行元素下标的选取
new_els=driver.find_elements_by_tag_name('input')
new_els[2].send_keys('12222222')
new_els[3].send_keys('123@qq.com')
#展示效果
time.sleep(3)
#退出浏览器
driver.quit()

























