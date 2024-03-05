#其实就是driver.find_element_by_name的运用
'''
1#导入模块
from selenium import webdriver
2#实例化浏览器对象
driver=webdriver.Chrome()
3#打开网页
driver.get('http://www.baidu.com')

#实现需求
@#输入定位用户名
username=driver.find_element_by_name('userA')
#输入定位密码
password=driver.find_element_by_name('passwordA')

'''
#如果当前页面中有多个元素，且元素的特征值是相同的，则默认会获取第一个
#符合要求的特证对应的元素
#因此，定义元素时需要保证使用的特征值可以代表目标元素在当前页面的唯一性



















































