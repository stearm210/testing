'''
一种随机生成的信息（数字、字母、汉字、图片、算术题）等
为了防止恶意的请求行为，增加安全性
自动化测试过程中，必须处理验证码，否则无法继续执行后续测试



#打开谷歌浏览器搜百度之后登录
然后右键检查，选择Application下的cookies中
的BDUSS键和后面一串值即可
BDUSS
puMDY3SkFFMFF-N2V5M2wwa1d4V1NyM1ZrRnFWaXdvTEZhTX5aOTh5QzJHaFZtSUFBQUFBJCQAAAAAAAAAAAEAAACN3Z2-uPSx2sDPxbfR9DExAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALaN7WW2je1lS1

之后组装成字典数据
{'name':BDUSS,value':puMDY3SkFFMFF-N2V5M2wwa1d4V1NyM1ZrRnFWaXdvTEZhTX5aOTh5QzJHaFZtSUFBQUFBJCQAAAAAAAAAAAEAAACN3Z2-uPSx2sDPxbfR9DExAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALaN7WW2je1lS1
}
'''

#cookie:绕过登录操作
from time import sleep, strftime
from selenium import webdriver
#实例化浏览器对象
driver=webdriver.Chrome()
#打开页面
driver.get('http://www.baidu.com')
driver.maximize_window()#窗口最大化
#cookie:绕过登录
#<1>整理关键cookie信息为字典数据
cookie_value={'name':'BDUSS',
              'value':'puMDY3SkFFMFF-N2V5M2wwa1d4V1NyM1ZrRnFWaXdvTEZhTX5aOTh5QzJHaFZtSUFBQUFBJCQAAAAAAAAAAAEAAACN3Z2-uPSx2sDPxbfR9DExAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALaN7WW2je1lS1'
}

#<2>调用方法，添加cookie信息
driver.add_cookie(cookie_value)

#<3>刷新页面，发送cookie信息给服务器进行验证
driver.refresh()

#注意
#1.本地浏览器中登录的账号不能退出，否则cookie信息过期
#那时就需要重新获取了
#2.不同项目能够进行登录功能绕过的cookie字段信息不一样，
#具体看情况问开发
#3.利用cookie绕过登录，则不能对登录功能本身进行测试
#4.个别项目如果想要绕过登录，可能需要添加多个cookie字段



#展示效果
sleep(3)
#退出浏览器
driver.quit()




























































































































































































































































