# import matplotlib
# print(matplotlib.matplotlib_fname())
import os
import chardet
f = open('test.txt','r')
s = f.read() #读取文件内容,如果是不识别的encoding格式（识别的encoding类型跟使用的系统有关），这里将读取失败
f.close()
'''假设文件保存时以gb2312编码保存'''
u = s.decode('gb2312') #以文件保存格式对内容进行解码，获得unicode字符串

'''下面我们就可以对内容进行各种编码的转换了'''
str = u.encode('utf-8')#转换为utf-8编码的字符串str
str1 = u.encode('gbk')#转换为gbk编码的字符串str1
str1 = u.encode('utf-16')#转换为utf-16编码的字符串str1
print(str1)