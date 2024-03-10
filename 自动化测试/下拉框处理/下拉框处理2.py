'''
使用注册A.html的页面，完成对城市的下拉框操作
1.选择广州
2.暂停2秒，选择上海
3.暂停2秒，选择北京
'''
from time import sleep
from selenium import webdriver
from selenium.webdriver.support.select import Select
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#定位下拉框元素
sel=driver.find_element_by_id('selectA')
#实例化下拉框选择对象
se=Select(sel)

#选择广州
#通过索引选择目标元素
se.select_by_index(2)
sleep(2)

#选择上海
#通过value属性值选择目标元素
se.select_by_value('sh')
sleep(2)

#选择北京
#通过可见文本信息选择目标元素
se.select_by_visible_text('A北京')
sleep(2)


#展示效果
sleep(3)
#退出浏览器
driver.quit()

'''
1.如果页面内需要操作的下拉框元素有多个，需要根据目标下拉框，
依次实例化下拉框选择对象
2.根据具体要求，三种下拉框内容元素选择方法，任意选择其中一种
'''







































































