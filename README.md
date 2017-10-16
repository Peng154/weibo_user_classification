# weibo_user_classification

## requirements
 - python-3.6
 - python-tensorflow
 - python-jieba
 - python-gensim
 - python-scikit-learn
 - python-numpy
 
## 简要说明
  1. 用户微博信息的收集使用八爪鱼工具，抓取每个用户最多300条微博
  2. 利用jieba分词工具将微博文本分词并且去停用词，然后利用gensim库对词向量进行tf-idf转换，最后特征提取成lsi模型
  具体细节可以参考 ： http://lib.csdn.net/article/machinelearning/42800 ，或者直接看代码。
  3. 人工对用户微博文本打标签
  4. 利用svm进行是否有购买意向的二分类和softmax进行具体有哪一类购买意向的分类
  5. 需要数据的可以联系我的邮箱：826113664@qq.com
