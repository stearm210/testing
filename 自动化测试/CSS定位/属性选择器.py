
'''
说明：根据元素的属性名和值来选择
格式：[attribute=value]  element[attribute=value]
例子：[type="password"]<选择type属性值为password的元素>


driver.find_element_by_css_selector('input[placeholder="请输入密码"]').send_keys('123456')


'''
'''
属性选择器的其他写法
1.标签名[属性名^="属性值开头部分内容"]:根据给出的属性值开头内容定位元素
driver.find_element_by_css_selector('[id^="pas"]').send_keys('123')

2.标签名

'''
















