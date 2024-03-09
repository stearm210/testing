from time import sleep
from selenium import webdriver
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

'''
#4.is_displayed()判断元素是否可见
#说明：此方法多用于对元素在页面内显示效果的判断时使用（元素不显示不意味着一定无法定位）
span=driver.find_element_by_name('sql')
print('目标元素是否显示：',span.is_displayed())


#5.is_enabled()判断元素是否可用


#6.is_selected()判断元素是否选中，用来检查复选框或者单选按钮是否选中


'''






sleep(3)
driver.quit()









































