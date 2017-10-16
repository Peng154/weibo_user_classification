import os, re, time, pickle, numpy.random
from collections import namedtuple
import jieba
from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import numpy as np

UserFrame = namedtuple('UserFrame',['str_list', 'uid', 'label'])

class WeiBoContent(object):

    def __init__(self, dataDir='./data'):
        self.dataDir = dataDir
        # 这个是爬虫扒下来的数据中有毒的一个字符
        self.errorStr = b'\xe2\x80\x8b'.decode()
        # 停用词表
        self.stopWords = self.getStopWords('{}/{}'.format(self.dataDir, 'stopwords.txt'))
        # 缓存目录
        self.cacheDir = './cache'
        # 选取的主题数量
        self.topics_no = 150
        if(not os.path.exists(self.cacheDir)):
            os.mkdir(self.cacheDir)

    def loadData(self, type, sub_classes):
        '''
        获取已经预处理好的数据
        :param type: 一共有'svm','soft_max','predict'三个种类,分别代表svm和soft_max的训练测试数据以及需要真正预测的数据
        :param sub_classes: 有‘train’和‘test’两类,分两个文件夹存放
        :return:返回对应的数据集
        '''
        if type == 'predict':
            sub_classes = 'test'

        file_path = "{a}/{b}/{c}.pkl".format(a=self.dataDir, b=sub_classes, c=type)

        if os.path.exists(file_path):
            file = open(file_path, 'rb')
            data =  pickle.load(file)
            file.close()
            return data
        else:
            raise ValueError("无法找到文件：{}！！".format(file_path))

    def getStopWords(self, filePath):
        """
        获取停用词表
        :param filePath:
        :return:
        """
        file = open(filePath, 'rb')
        file.seek(0)
        # 需要排除\r\n。。。。
        stopWords = [line.decode('utf-8')[:-2] for line in file]
        file.close()
        return stopWords

    def clearWords(self, str_):
        '''
        去掉无用词语
        :param str_:
        :return:
        '''
        str = str_.replace('地址：O网页链接', '')
        str = str.replace ('O网页链接','')
        str = str.replace ('转发微博','')
        return str

    def convert_doc_to_wordlist(self, doc):
        """
        把文档转换成词语，并去停用词
        :param doc:
        :return:
        """
        word_list = jieba.lcut(doc, cut_all=False)
        word_list = [word for word in word_list if word not in self.stopWords]
        return word_list

    def getDictionary(self, words_list, dict_path, useCache=True):
        """
        生成词典，并且缓存起来
        :param doc_list:
        :param dict_path:
        :param useCache:
        :return:
        """
        # 如果以前有数据，而且使用缓存，直接返回
        if(os.path.exists(dict_path)):
            if(useCache):
                print("使用缓存词典：{}".format(dict_path))
                return corpora.Dictionary.load(dict_path)

        # 重新加载
        print("重新生成词典")
        dictionary = corpora.Dictionary()
        for words in words_list:
            dictionary.add_documents([words])

        # 去掉出现次数太少的词，这些词没有代表性
        small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 5]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()

        # 词典缓存起来
        if useCache:
            dictionary.save(dict_path)

        print("===词典已经生成===")
        return dictionary

    def getTfIdfModel(self, model_path, dictionary=None, useCache=True):
        '''
        获取tf-idf模型，用于构造tf-idf向量
        :param model_path: model缓存路径
        :param dictionary: 构造所用字典，当使用缓存的时候，可以为空
        :param useCache: 是否使用缓存
        :return:
        '''
        if os.path.exists(model_path):
            if useCache:
                file = open(model_path, 'rb')
                tfidf_model = pickle.load(file)
                file.close()
                print("使用已有缓存tf-idf模型：{}".format(model_path))
                return tfidf_model

        if dictionary is None:
            raise ValueError('字典为空，且没有缓存，无法生成tf-idf模型')
        else:
            print("重新生成tf-idf模型。。。。")
            tfidf_model = models.TfidfModel(dictionary=dictionary)
            if useCache:
                file = open(model_path,'wb')
                pickle.dump(tfidf_model, file=file)
                file.close()
                print("缓存tf-idf模型到文件：{}".format(model_path))
            return tfidf_model

    def getLsiModel(self, model_path, corpus=None, dictionary=None, num_topics=50, useCache=True):
        '''
        获取lsi模型，用于构造lsi向量
        :param model_path: model缓存路径
        :param corpus: 构造所用tf-idf向量列表，当使用缓存的时候，可以为空
        :param dictionary: 构造所用字典，当使用缓存的时候，可以为空
        :param num_topics: 主题数量？？？
        :param useCache: 是否使用缓存
        :return:
        '''
        if os.path.exists(model_path):
            if useCache:
                file = open(model_path, 'rb')
                lsi_model = pickle.load(file)
                file.close()
                print("使用已有缓存lsi模型：{}".format(model_path))
                return lsi_model

        if corpus is None or dictionary is None:
            raise ValueError('未找到缓存，并且没有字典或者tf-idf集合，无法生成lsi模型')
        else:
            print("重新生成lsi模型。。。。")
            lsi_model = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
            if useCache:
                file = open(model_path,'wb')
                pickle.dump(lsi_model, file=file)
                file.close()
                print("缓存lsi模型到文件：{}".format(model_path))
            return lsi_model

    def preProcessData(self, rawFiles, types, outputFile, train_data_rate=0.8, minWeiBoNo=20, useCache=True):
        """
        对数据进行预处理，rawFiles是用户微博数据文件列表，types是每个文件的标签，最后输出到一个缓存文件中，
        存储的是经过预处理、向量特征化之后的向量数据以及相关用数据。
        数据预处理步骤：
        1. 分词，去停用词以及无关的词，比如转发微博，网址链接等，还需要去除出现次数过少的词
        2. tf-idf向量统计
        3. LSI模型提取可能隐含的主题，向量降维到 self.topics_no 的维数
        4. 根据分配比例保存到文件中
        :param rawFiles:用户微博数据文件列表
        :param types: 文件对应的标签
        :param outputFile: 输出缓存文件名
        :param train_data_rate: 数据中用于训练的数据比例
        :param minWeiBoNo: 单个用户最小微博数量
        :param useCache: 是否缓存
        :return:
        """

        dict_path = "{}/words_dict.dict".format(self.cacheDir)
        tfidf_model_path = "{}/{}".format(self.cacheDir, 'tfidf_model.pkl')
        lsi_model_path = "{}/{}".format(self.cacheDir, 'lsi_model.pkl')
        train_output_file = "{}/{}/{}".format(self.dataDir, 'train', outputFile)
        test_output_file = "{}/{}/{}".format(self.dataDir, 'test', outputFile)
        pattern = re.compile(r"[\s+－\-\|️ ×丨“”：:\.\!\/_,$%^*()【】\[\]\"\']+|[+—！{／}<↓>《》/\\，。？\?、~@#￥%…&*（）]+")
        users = []
        uid_set = set()

        # 判断是否需要预处理数据
        if(train_data_rate!=0 and os.path.exists(train_output_file)):
            if useCache:
                return
        elif(train_data_rate == 0 and os.path.exists(test_output_file)):
            if useCache:
                return

        for i in range(len(rawFiles)):
            filename = '{}/{}'.format(self.dataDir, rawFiles[i])
            file = open(filename, encoding='utf-8')

            last_uid = ''
            str_list = []

            # 对于每一个文件，都是一种类别
            for line in file:
                # 去掉换行符
                line = line[:-1]
                splits = line.split(',')
                uid_set.add(splits[0])
                length = len(splits)
                if(length==3):
                    cur_uid = splits[0]
                    str = self.clearWords(splits[-1])
                    # 如果用户不同
                    if(cur_uid != last_uid):
                        # 不是第一条以及大于最小微博数
                        if(last_uid != '' and len(str_list) > minWeiBoNo):
                            users.append(UserFrame(str_list, last_uid, types[i]))
                        last_uid = cur_uid
                        str_list = []
                        if str != '':
                            str_list.append(str)
                    # 如果两次用户相同，直接添加语言列表
                    else:
                        if str != '':
                            str_list.append(str)

            file.close()

        print("用户总数量：{}".format(len(uid_set)))
        print("合格用户数量：{}".format(len(users)))
        # print(users[101])

        # 每个元素都是每个用户分好词的列表
        words_list = []
        uid_list = []
        label_list = []

        # 分词，去停用词
        print("开始分词")
        start = time.time()
        for user in users:
            str = ','.join(user.str_list)
            str = re.sub(pattern, '', str)
            seg_list = jieba.lcut(str)

            words_list.append(seg_list)
            uid_list.append(user.uid)
            label_list.append(user.label)
        end = time.time()
        print("分词结束，用时：{}s".format(end-start))

        # -------------------- 开始将词语转换成tf-idf向量表示-------------------------
        print("转换成tf-idf向量")
        # tf-idf列表
        tfidf_list = []
        # 词典
        dictionary = self.getDictionary(words_list, dict_path)
        tfidf_model = self.getTfIdfModel(tfidf_model_path, dictionary=dictionary, useCache=True)

        for words in words_list:
            # 转换成词频向量表示
            word_bow = dictionary.doc2bow(words)
            # 转换成tf-idf向量表示
            tfidf_list.append(tfidf_model[word_bow])

        # -------------------- 开始将tf-idf转化成lsi-------------------------
        # lsi模型，50个主题？？？？
        lsi_model = self.getLsiModel(lsi_model_path, tfidf_list, dictionary, num_topics=self.topics_no, useCache=True)
        corpus_lsi = [lsi_model[doc] for doc in tfidf_list]

        # -------------gensim的矩阵到sklearn矩阵数据结构的转换----------------
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_lsi:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        lsi_sparse_matrix = csr_matrix((data, (rows, cols)))  # 稀疏向量
        lsi_matrix = lsi_sparse_matrix.toarray()  # 密集向量


        # 训练集合测试集的分割点
        idxs = [idx for idx in range(lsi_matrix.shape[0])]
        numpy.random.shuffle(idxs)
        seg_pos = int(float(lsi_matrix.shape[0])*train_data_rate)
        print("分割点：{}".format(seg_pos))
        train_idxs = idxs[0:seg_pos]
        test_idxs =  idxs[seg_pos:]

        print("购买样本数：{}".format(sum(label_list)))

        if seg_pos != 0:
            print("保存训练数据到文件：{}".format(train_output_file))
            file = open(train_output_file, 'wb')
            train_vec = [lsi_matrix[i] for i in train_idxs]
            train_vec = np.array(train_vec)
            train_vec = train_vec.reshape((train_vec.shape[0], -1))
            train_uid = [uid_list[i] for i in train_idxs]
            train_label = [label_list[i] for i in train_idxs]
            pickle.dump((train_vec, train_uid, train_label), file)
            file.close()

        if seg_pos!=lsi_matrix.shape[0]:
            print("保存测试数据到文件：{}".format(test_output_file))
            file = open(test_output_file, 'wb')
            test_vec = [lsi_matrix[i] for i in test_idxs]
            test_vec = np.array(test_vec)
            test_vec = test_vec.reshape((test_vec.shape[0], -1))
            test_uid = [uid_list[i] for i in test_idxs]
            test_label = [label_list[i] for i in test_idxs]
            pickle.dump((test_vec, test_uid, test_label), file)
            file.close()

    def mergeFiles(self, fileList, outputFile_, useCache=True, relative_path=True):
        """
        合并列表中的文件,并且去掉一些没有用的特殊字符
        :param fileList:需要合并的文件列表
        :param outputFile:需要输出的文件列表
        :param useCache: 是否使用以前的缓存
        :param relative_path:是否相对路径，相对路径就是'./data'目录下
        :return:
        """

        outFilePath = "{}/{}".format(self.dataDir,outputFile_)

        if os.path.exists(outFilePath):
            if useCache:
               return

        print("开始合成文件：{}".format(outFilePath))

        # 输出文件
        outputFile = open(outFilePath, 'w', encoding='utf-8')

        for filePath in fileList:
            if relative_path:
                file = open("{}/{}".format(self.dataDir, filePath), encoding='utf-16LE')
            else:
                file = open(filePath, encoding='utf-16LE')

            line_no = 0

            for line in file:

                line = line[1:-2]
                # 去除有毒的字符
                line = line.replace(self.errorStr, '')

                # 去掉第一行无用字符
                if line_no == 0 and line == '"用户名","时间","字段3':
                    print("丢弃字符串：",line)
                else:
                    outputFile.write(line+'\n')
                    # print(line)
                line_no += 1

            file.close()

        outputFile.close()

def test():
    file = open('./data/train/20170615.pkl', 'rb')
    train_data = pickle.load(file)
    file.close()
    file = open('./data/test/20170615.pkl', 'rb')
    test_data = pickle.load(file)
    file.close()

    tuned_parameters = [{'kernel':['rbf'], 'gamma':[1e-2, 1e-1, 1.0, 2.0, 4.0, 8.0, 10.0],
                         'C':[0.5, 1, 2, 5, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                       scoring='precision_weighted')
    clf.fit(train_data[0], train_data[2])

    print(clf.best_params_)
    str = "{'kernel': 'linear', 'C': 10}"

    pre_result = clf.predict(test_data[0])
    print(test_data[0].shape)
    # print(pre_result)
    # print(classification_report(test_data[2], pre_result))
    # print(np.sum(pre_result==test_data[2])/pre_result.shape[0])
    # print("标签：")
    # print(test_data[2])
    # print("预测：")
    # print(list(pre_result))

