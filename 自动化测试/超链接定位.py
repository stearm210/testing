'''
link_text方法
#实现需求
1.使用link_text定位(访问xx网站超链接)，并点击
#点击方法，元素对象.click()
#link_text方法：此方法只针对超链接元素(a元素)
#并且需要输入超链接的全部文本信息(当页面中有'访问新浪网站几个字时')
driver.find_element_by_link_text('访问新浪网站').click()

#注意，虽然是传入部分的文本信息，但是需要确定唯一性才可以使用
driver.find_element_by_partial_link_text('新浪').click()


'''





































