'''
层级选择器根据元素的父子来选择
格式1：element1>element2    通过leement1来定位element2，并且element2必须为element1的直接子元素
格式2：element1 element2    通过element1来定位element2，并且element2为element1的后代元素
'''

'''
1.父子层级关系：夫层级>子层级策略
driver.find_element_by_css_selector('#pa>input').send_keys('admin')

2.祖辈后代层级关系：祖辈策略 后代策略(这里的后代用属性选择器)
driver.find_element_by_css_selector('form [placeholder="请输入用户名"]').send_keys('admin')

#展示效果
sleep(3)

#退出浏览器
driver.quit()

'''



























