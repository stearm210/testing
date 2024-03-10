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
'''
自定义弹窗
自定义弹窗可以通过标签、属性等进行定位，到时候直接进行定位处理即可

'''


#展示效果
sleep(3)
#退出浏览器
driver.quit()




















































































































