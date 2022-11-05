#百度翻译爬虫
import requests
import json
if __name__=="__main__":
    url='https://fanyi.baidu.com/sug'
    #post请求参数处理（同get请求一致）
    world=input('enter a world:')
    data={'kw':world}
    #UA
    headers={'User-Agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Mobile Safari/537.36 Edg/106.0.1370.52'}
    response=requests.post(url=url,data=data,headers=headers)
    #json()返回的是obj（确认响应数据是json类型，在响应头的Content-Type查看）
    obj=response.json()
    print(obj)
    #储存
    with open('./requests2.json','w',encoding='utf-8') as fp:
        json.dump(obj,fp=fp,ensure_ascii=False)#中文不能使用ascii编码

    print('over')