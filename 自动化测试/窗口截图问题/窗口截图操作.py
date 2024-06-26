'''
实现窗口截图操作
'''
from time import sleep, strftime
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#打开页面，完成操作
#1.注册信息
driver.find_element_by_id('userA').send_keys('admin')
driver.find_element_by_id('passwordA').send_keys('123456')
#2.截图保存(里面填写路径)
driver.get_screenshot_as_file('/info.png')

#拓展，使用时间戳防止文件重名被覆盖
#说明：windows系统文件名不支持特殊字符，尽量只使用下划线
now_time=strftime('%Y%ms_%H%M%')
driver.get_screenshot_as_file('/info.png')

#拓展：给元素截图
#使用元素标签找到对应元素
btn=driver.find_element_by_tag_name('button')
btn.screenshot('./btn.png')

#展示效果
sleep(3)
#退出浏览器
driver.quit()






































































































































































































































