'''
如何实现多窗口切换：
使用句柄可以快速实现切换
在selenium中封装了当前窗口句柄、获取所有窗口句柄和切换到指定句柄窗口的办法
句柄为：handle，窗口的唯一识别码

如果当前页面操作完了之后，需要回到原始页面进行操作
则需要回到原始页面操作才可以继续填写原始页面信息

'''
from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')

#需求：打开页面
#1.点击注册页面链接
driver.find_element_by_id('ZCA').click()

#当前窗口句柄
#浏览器的任意一个生命周期
#任意的一个窗口都有唯一的一个句柄值，
# 可以通过句柄值完成窗口的切换操作
print('当前句柄值：',driver.current_window_handle)

#切换窗口操作
#<1>获取所含有窗口句柄（包含新窗口）
handles=driver.window_handles
print(handles)
print(type(handles))
#<2>切换窗口：列表的-1索引对应的值，始终是最新窗口的句柄值
driver.switch_to.window(handles[-1])



#2.在打开的页面中，填写注册信息
driver.find_element_by_id('userA').send_keys('admin')
driver.find_element_by_id('passwordA').send_keys('123456')

sleep(2)

#如果想要操作原始页面，则需要完成窗口切换之后才能
#填写原始页面信息
driver.find_element_by_id('user').send_keys('admin')
driver.find_element_by_id('password').send_keys('123456')


#展示效果
sleep(3)
#退出浏览器
driver.quit()






































































































































































































































