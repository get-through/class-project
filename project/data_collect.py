import requests
import re
import json

def get_page(url):
	#采集器函数
	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'}
	response = requests.get(url, headers = headers)
    if response.status_code == 200:
        return response.text
    else:
		return 'GET HTML ERROR ! '


def parse_page(html):
	#解析器函数
	pattern = re.compile('<em class="">(.*?)</em>.*?<a href="(.*?)" rel="external nofollow" rel="external nofollow" >.*?<span class="title">(.*?)</span>.*?<div class="bd">.*?<p class="">(.*?)&nbsp.*?<br>(.*?)&nbsp;/&nbsp;(.*?)&nbsp;/&nbsp;(.*?)</p>.*?<span class="rating_num".*?"v:average">(.*?)</span>' , re.S)
	items = re.findall(pattern , html)
	for item in items:
		yield {
		  'rank': item[0],
		  'href': item[1],
		  'name': item[2],
		  'director': item[3].strip()[4:],
		  'year': item[4].strip(),
		  'country': item[5].strip(),
		  'style': item[6].strip(),
		  'score': item[7].strip()
		}


def write_to_file(content):
	#写入文件函数
	with open('result.txt' , 'a' , encoding = 'utf-8') as file:
		file.write(json.dumps(content , ensure_ascii = False) + '\n')


if __name__== "__main__":
	# 主程序
	for i in range(10):
		url= 'https://movie.douban.com/top250?start='+ str(i*25)+ '&filter'
		for res in parse_page(get_page(url)):
			write_to_file(res)