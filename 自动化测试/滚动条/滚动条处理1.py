
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
#打开注册页面A，暂停2秒之后，滚动条落在最底层，暂停2秒之后，恢复原位
sleep(2)
#1.js代码
js_down="windows.scrollTo(0,1000)"
#2.执行js代码（向下代码）
driver.execute_script(js_down)
#3.向上代码
#反向向上则需要将坐标归0
js_up="window.scrollTo(0,0)"
driver.execute_script(js_up)



#展示效果
sleep(3)
#退出浏览器
driver.quit()
































































































































































