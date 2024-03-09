
from time import sleep
from selenium import webdriver
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

#1.最大化窗口maximize_window()模拟浏览器最大化按钮
'''
如果能够在打开页面时，全屏显示页面，就能尽最大可能加载更多页面元素
提高可定位性
'''
#driver.maximize_window()

#2.设置浏览器窗口，set_window_size(width,height)设置浏览器宽、高（像素点）
#driver.set_window_size(300,300)
#上面用于查看页面是否可以自适应在web和app端切换使用

#3.set_windows_position(x,y)设置浏览器窗口位置，设置浏览器位置
driver.set_window_position(300,300)


sleep(3)
driver.quit()






















