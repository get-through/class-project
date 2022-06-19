chinese_hmm.py实现中文分词和词性标注，english_hmm.py实现了英文词性标注
使用的conda环境导出在env.yaml中

test.words是WSJ_24的测试文字

直接在终端输入即可分词
将输入取消直接添加输入文件到 _, prep = preprocess(vocab, "")语句的引号中也可以
（需要每行一个英文词或汉字，且每两句句子间有空行的格式）
