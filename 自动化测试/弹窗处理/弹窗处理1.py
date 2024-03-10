'''
使用注册A.html的页面，完成对城市的下拉框操作
1.选择广州
2.暂停2秒，选择上海
3.暂停2秒，选择北京
'''

from time import sleep
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('www.baidu.com')


#对应处理
#1.点击alert按钮
driver.find_element_by_id('alerta').click()#警告框
driver.find_element_by_id('confirma').click()#确认框
#2.关闭警告框
#先切换到弹窗
alert=driver.switch_to.alert
#获取弹窗信息
print('弹窗信息是：',alert.text)
#去除弹窗（同意/移除）
sleep(2)
alert.accept()
alert.dismiss()

#3.输入用户名admin
driver.find_element_by_id('userA').end_keys('admin')



#展示效果
sleep(3)
#退出浏览器
driver.quit()










































































