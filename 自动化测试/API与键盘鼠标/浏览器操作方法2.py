from time import sleep
from selenium import webdriver
#实例化对象
driver=webdriver.Chrome()
driver.get('www.baidu.com')

#关闭浏览器页面，模拟点击浏览器关闭按钮



#关闭浏览器驱动对象，关闭所有程序启动的窗口


#title 获取页面title

#current_url 获取当前页面的URL

'''
在没有实现浏览器页面切换操作前，close()方法关闭的是原始页面
场景：关闭单个页面使用
driver.close()
'''

'''
使用场景：浏览器的标题和URL地址属性，可以用来做断言使用
print('关闭前页面标题：',driver.title)
print('关闭前页面地址：',driver.current_url)
'''




sleep(3)
driver.quit()





















