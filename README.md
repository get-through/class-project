# class-project

test-hmm.py 可以实现动态输入多行英文，可以处理标点，然后输出句子和对应好的tag
我还没做几个矩阵的存储，存了的话就确实是直接可以用了
utils_pos.py 是辅助 test-hmm.py 和 hmm.py 的

hmm.py 是 test-hmm.py 的原代码，有更多注释，和 https://blog.csdn.net/weixin_43093481/article/details/115253958 的是一样的，这篇文章的代码在https://gitcode.net/mirrors/Ogmx/Natural-Language-Processing-Specialization?utm_source=csdn_github_accelerator ，但应该没必要下载，和hmm.py里是一样的

for_attempts.py 里有我自己乱写的可以用python的工具包生成数据集的代码，nltk可以生成英文数据集，jieba可以对中文分词，中文tagging的还没做，两个都要又配一些环境，你可以不装那些环境，需要扩大数据集的话我这里处理好再上传就可以了

我不会爬虫 或许你会的话可以写一下爬虫爬新闻/博客的文章 当然直接人工复制黏贴也可以
