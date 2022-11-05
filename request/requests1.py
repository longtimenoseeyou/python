#网页采集器（爬虫）
#UA:User-Agent(请求载体的身份标识)
import  requests
if __name__ == "__main__":
    #step1:url
    headers={'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Mobile Safari/537.36 Edg/106.0.1370.52'}
    url='https://www.baidu.com/s'
    #step2:发起请求
    kw=input('enter a world:')#将要搜索的内容封装到字典中
    param ={'wd':kw}
    response=requests.get(url=url,params=param,headers=headers)
    #step3:获取响应数据
    print(response.text)
    #持久化储存
    with open("./baidu.html",'w',encoding='utf-8') as fp:
        fp.write(response.text)
    print('爬虫结束')
