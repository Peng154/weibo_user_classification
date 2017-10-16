import tensorflow as tf
from WeiBoContent import WeiBoContent,UserFrame
import os, random, numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle
from socket import *
import matplotlib.pyplot as plt

class Classifier(object):

    def __init__(self, weight_decay=0.01, batchSize=128):
        print("开始预处理数据。。。。")
        self.weibo = WeiBoContent()
        self.svm_dataPath = 'svm.pkl'
        self.softmax_dataPath = 'softmax.pkl'
        self.modelDir = './cache/model'
        # 训练集大小
        self.batchSize = batchSize
        self.weight_decay = weight_decay
        # 分类的数量
        self.class_no = 4

        if not os.path.exists(self.modelDir):
            os.mkdir(self.modelDir)

        # --------------合并同一类数据集----------------
        self.weibo.mergeFiles(['人民网(0-141).csv', '人民网（141-541）.csv', '人民网（542-1041）.csv',
                               '人民网（1041-1183）.csv', '人民网（1183-1601）.csv', '人民网（1601-1759）.csv'
                                  , '人民网（1759-1941）.csv', '人民网（2000-2442）.csv',
                               '人民网（2443-2500）.csv'], '人民网.csv', useCache=True)

        self.weibo.mergeFiles(['母婴(0-400).csv', '母婴(400-500).csv', '母婴（500-606）.csv'
                                  , '母婴（607-809）.csv', '母婴(810-1000).csv'], '母婴.csv', useCache=True)

        self.weibo.mergeFiles(['比利时孕妇装（0-53）.csv', '比利时孕妇装（54-200）.csv',
                               '比利时孕妇装（201-263）.csv', '比利时孕妇装（264-500）.csv',
                               '比利时孕妇装（501-700）.csv' ,'帮宝适（0-200）.csv','帮宝适（201-400）.csv'
                                  ,'帮宝适（401-550）.csv'], '衣.csv', useCache=True)

        self.weibo.mergeFiles(['美素佳儿0-220.csv', '美素佳儿221-401.csv',
                               '美素佳儿406-550.csv'], '食.csv', useCache=True)

        self.weibo.mergeFiles(['ABC婴儿车.csv', '圣得贝婴儿车.csv'], '行.csv', useCache=True)

        self.weibo.mergeFiles(['乐然婴儿床.csv'], '住.csv', useCache=True)


        svm_rawFiles = ['人民网.csv', '母婴.csv', '衣.csv','食.csv', '住.csv', '行.csv']
        svm_labels = [0, 1, 1, 1, 1, 1]
        softmax_rawFiles = ['衣.csv', '食.csv', '住.csv', '行.csv']
        softmax_labels = [0, 1, 2, 3]

        # -------------数据预处理，向量转换-------------
        self.weibo.preProcessData(svm_rawFiles, svm_labels, self.svm_dataPath, minWeiBoNo=40)
        self.weibo.preProcessData(softmax_rawFiles, softmax_labels, self.softmax_dataPath, minWeiBoNo=40)

        print("数据预处理结束。。。。")


    def weight_variable(self, shape, wd, stddev):
        """
        返回权重矩阵
        :param wd:
        :param stddev:
        :return:
        """
        initial = tf.truncated_normal(shape=shape, stddev=stddev)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(initial), wd)
            tf.add_to_collection('losses', weight_decay)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """
        返回偏置矩阵
        :return:
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nfromm(self, m, n, unique=True):
        """
        从[0, m)中产生n个随机数
        :param m:
        :param n:
        :param unique:
        :return:
        """
        if unique:
            box = [i for i in range(m)]
            out = []
            for i in range(n):
                index = random.randint(0, m - i - 1)

                # 将选中的元素插入的输出结果列表中
                out.append(box[index])

                # 元素交换，将选中的元素换到最后，然后在前面的元素中继续进行随机选择。
                box[index], box[m - i - 1] = box[m - i - 1], box[index]
            return out
        else:
            # 允许重复
            out = []
            for _ in range(n):
                out.append(random.randint(0, m - 1))
            return out

    def getNextBatch(self, type):
        """
        获取一个batch数据集或者test数据集
        :param type:
        :return:
        """
        if not (hasattr(self, 'softmax_traindata') or hasattr(self, 'softmax_testdata')):
            raise RuntimeError('还未加载soft_max训练和测试数据！！')

        if type == 'train':
            data_num = self.softmax_traindata[0].shape[0]
            idxs = self.nfromm(data_num, self.batchSize)
            batchData = []
            batchLabel = []
            for i in idxs:
                batchData.append(self.softmax_traindata[0][i])
                batchLabel.append(self.softmax_traindata[2][i])

            batchData = np.array(batchData)
            batchData = batchData.reshape([self.batchSize, -1])

            return batchData, batchLabel

        elif type == 'test':
            return self.softmax_testdata[0], self.softmax_testdata[2]

    def inference(self, inputs):
        """
        构建softmax网络结构
        :param inputs: 预测结果
        :return:
        """
        dim = self.weibo.topics_no

        with tf.variable_scope('full1') as scope:
            weights = self.weight_variable([dim, 256], self.weight_decay, stddev=(2. / (dim)) ** 0.5)
            bias = self.bias_variable([256])
            pre_activation = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
            local1 = tf.nn.relu(pre_activation)

        with tf.variable_scope('full2') as scope:
            weights = self.weight_variable([256, 256], self.weight_decay, stddev=(2. / 256)**0.5)
            bias = self.bias_variable([256])
            pre_activation = tf.nn.bias_add(tf.matmul(local1, weights), bias)
            local2 = tf.nn.relu(pre_activation)

        with tf.variable_scope('softmax_linear') as scope:
            weights = self.weight_variable([256, self.class_no], self.weight_decay, stddev=(2. / 256) ** 0.5)
            bias = self.bias_variable([self.class_no])
            softmax_linear = tf.nn.bias_add(tf.matmul(local2, weights), bias)

        return softmax_linear

    def train_softmax(self, saveModel=True):
        """
        训练soft_max分类器
        :param saveModel: 是否保存模型
        :return:
        """

        if tf.gfile.Exists('{}/softmax.ckpt.meta'.format(self.modelDir)):
            if saveModel:
                print('已存在训练好的softmax模型，停止训练。')
                return

        print("开始训练softmax分类器")
        # 训练参数
        steps = 6000
        lr = 3e-5 # 步长

        # 记录损失和训练准确率，用于画图
        costs = []
        accuracies = []

        # 加载训练数据
        self.softmax_traindata = self.weibo.loadData('softmax', 'train')

        # softmax 的输入,以及标签
        inputs = tf.placeholder(tf.float32, [self.batchSize, self.weibo.topics_no])
        labels = tf.placeholder(tf.int64, [self.batchSize])

        y = self.inference(inputs=inputs)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y)

        # L2正则化
        loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', loss)

        total_loss = tf.add_n(tf.get_collection('losses'))
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(total_loss)
        # 准确率
        correct_prediction = tf.equal(labels, tf.argmax(y ,axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 构建会话
        sess = tf.Session()
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        for step in range(steps):
            train_inputs, train_labels = self.getNextBatch('train')
            if step%10==0:
                train_accuracy = accuracy.eval(session = sess, feed_dict={inputs: train_inputs, labels:train_labels})
                train_loss = total_loss.eval(session =sess, feed_dict={inputs: train_inputs, labels:train_labels})
                costs.append(train_loss)
                accuracies.append(train_accuracy)
                print("step %d, training accuracy %g, loss %g" % (step, train_accuracy, train_loss))
            sess.run(train_op, feed_dict={inputs: train_inputs, labels:train_labels})

        # 画出损失函数变化图像
        fig = plt.figure()
        plt.semilogy(costs)
        plt.show(block=False)
        fig.savefig('./costs.png')
        # 画出准确率变化图像
        fig = plt.figure()
        plt.semilogy(accuracies)
        plt.show(block=False)
        fig.savefig('./accuracies.png')

        # 保存模型
        if saveModel:
            saver.save(sess=sess, save_path='{}/{}'.format(self.modelDir, 'softmax.ckpt'))

        # 清空训练数据，释放内存
        del self.softmax_traindata

    def test_softmax(self):
        # 加载测试数据
        self.softmax_testdata = self.weibo.loadData('softmax', 'test')
        # 测试一下准确率
        test_inputs, test_labels = self.getNextBatch('test')
        nums = test_inputs.shape[0]
        print("测试集总数：{}".format(nums))

        # softmax 的输入,以及标签
        inputs = tf.placeholder(tf.float32, [nums, self.weibo.topics_no])
        labels = tf.placeholder(tf.int64, [nums])

        y = self.inference(inputs=inputs)
        # 准确率
        correct_prediction = tf.equal(labels, tf.argmax(y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, save_path='{}/{}'.format(self.modelDir, 'softmax.ckpt'))
            test_accuracy = accuracy.eval(session=sess, feed_dict={inputs: test_inputs, labels:test_labels})
            result_matrix = y.eval(session=sess, feed_dict={inputs: test_inputs, labels:test_labels})
            print("测试准确率：{}".format(test_accuracy))
            print(result_matrix)

            sess.close()

    def train_svm(self, saveModel=True):
        train_data = self.weibo.loadData(type='svm', sub_classes='train')
        test_data = self.weibo.loadData(type='svm', sub_classes='test')

        # -------------网格搜索，寻找最优SVM参数---------------
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-1, 1.0, 2.0, 4.0, 8.0, 10.0],
        #                      'C': [0.5, 1, 2, 5, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        #
        # clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
        #                    scoring='precision_weighted')
        # clf.fit(train_data[0], train_data[2])
        #
        # print(clf.best_params_)
        # str = "{'C': 10, 'gamma': 2.0, 'kernel': 'rbf'}"
        #
        # pre_result = clf.predict(test_data[0])
        # print(test_data[0].shape)
        # print(np.sum(pre_result==test_data[2])/pre_result.shape[0])
        # --------------------------------------------------

        # 选择网格搜索找到的合适的参数
        clf = svm.SVC(C=10, kernel='rbf', gamma=2.0)
        # 训练
        clf.fit(train_data[0], train_data[2])
        pre_result = clf.predict(test_data[0])
        print(test_data[0].shape)
        print(np.sum(pre_result==test_data[2])/pre_result.shape[0])
        #
        # pre_result = clf.predict(train_data[0])
        # print(train_data[0].shape)
        # print(np.sum(pre_result == train_data[2]) / pre_result.shape[0])

        print("标签：")
        print(test_data[2])
        print("预测：")
        print(list(pre_result))

        # 保存模型
        if saveModel:
            svm_model_path = '{}/{}'.format(self.modelDir, 'svm_model.pkl')
            file = open(svm_model_path, 'wb')
            pickle.dump(file=file, obj=clf)
            file.close()
            print("保存svm模型到：{}".format(svm_model_path))

    def predict(self, csv_data_path):
        """
        传入csv文件路径，进行分类
        :param csv_data_path:
        :return:
        """
        self.weibo.mergeFiles([csv_data_path], 'predict.csv', useCache=False, relative_path=False)
        self.weibo.preProcessData(['predict.csv'],[-1],'predict.pkl',train_data_rate=0, minWeiBoNo=40, useCache=False)

        predict_data = self.weibo.loadData('predict', None)

        # 加载SVM模型
        svm_model_path = '{}/{}'.format(self.modelDir, 'svm_model.pkl')
        file = open(svm_model_path, 'rb')
        svm_clf = pickle.load(file)
        file.close()

        buy_users = []
        no_buy_users = []

        # 先利用SVM分类是否有购买倾向
        pre_result = svm_clf.predict(predict_data[0])
        # 分离购买与非购买用户
        for i in range(predict_data[0].shape[0]):
            if pre_result[i] == 1:
                buy_users.append(UserFrame(predict_data[0][i], predict_data[1][i], pre_result[i]))
            else:
                no_buy_users.append(UserFrame(predict_data[0][i], predict_data[1][i], pre_result[i]))

        # 预测购买的用户的特征矩阵
        buy_matix = []
        for user in buy_users:
            buy_matix.append(user.str_list)
        buy_matix = np.array(buy_matix)
        print(buy_matix.shape)
        # 然后针对有购买倾向的用户进行细化分类-------------
        # 如果有购买倾向的用户数不为0
        if len(buy_users)!=0:
            if not hasattr(self, 'sess'):
                # softmax 的输入,以及标签
                self.inputs = tf.placeholder(tf.float32, [1, self.weibo.topics_no])

                self.y = self.inference(inputs=self.inputs)
                saver = tf.train.Saver()
                self.sess = tf.Session()

                # 加载训练好的模型
                saver.restore(self.sess, save_path='{}/{}'.format(self.modelDir, 'softmax.ckpt'))

            result_matrix = []
            for i in range(buy_matix.shape[0]):
                result_matrix.append(self.y.eval(session=self.sess, feed_dict={self.inputs: [buy_matix[i]]}))

            result_matrix = np.array(result_matrix).reshape([len(result_matrix), -1])
            print(result_matrix.shape)
            print(np.argmax(result_matrix, 1))
            # 购买概率矩阵
            p_matrix = np.exp(result_matrix)
            for i in range(p_matrix.shape[0]):
                p_matrix[i] /= np.sum(p_matrix[i])
                # print(np.argmax(p_matrix, 1))
                # print(p_matrix)

            # 写入结果文件
            result_path = './cache/result.txt'
            file = open(result_path, 'w', encoding='utf-8')
            file.write('[')
            for i in range(len(no_buy_users)):
                file.write('{c}"uid":"{a}","buy":"{b}","clo":"0","eat":"0","live":"0","mov":"0"{d}'.format(
                    c='{', d='}',
                    a=no_buy_users[i].uid[:-1],
                    b=no_buy_users[i].label))
                if i != len(no_buy_users) - 1:
                    file.write(',')
                elif len(buy_users) != 0:
                    file.write(',')

            for i in range(len(buy_users)):
                file.write('{e}"uid":"{id}","buy":"{c}","clo":"{p1}","eat":"{p2}","live":"{p3}","mov":"{p4}"{f}'.format(
                    id=buy_users[i].uid[:-1], c=buy_users[i].label,
                    p1=p_matrix[i][0], p2=p_matrix[i][1],
                    p3=p_matrix[i][2], p4=p_matrix[i][3],
                    e='{', f='}'))
                if i != len(buy_users) - 1:
                    file.write(',')

            file.write(']')
            file.close()
            return result_path


if __name__ == '__main__':
    clf = Classifier(weight_decay=0.01, batchSize=128)
    clf.train_svm()
    # clf.train_softmax(saveModel=True)
    # clf.test_softmax()

    # 与服务器后台进行tcp通讯
    HOST = ''
    PORT = 21567
    ADDR = (HOST, PORT)
    BUFSIZE = 1024

    tcpSrvSock = socket(AF_INET, SOCK_STREAM)
    tcpSrvSock.bind(ADDR)
    tcpSrvSock.listen(5)

    while True:
        print('等待连接。。。。')
        tcpCliSock, addr = tcpSrvSock.accept()
        print('来自{}'.format(addr))
        # 接收并处理
        data = tcpCliSock.recv(BUFSIZE)
        print(data)
        print("data:{}".format(data.decode('utf-8')))
        path = data[:-2].decode('utf-8')
        if not os.path.exists(path):
            print("请求文件路径不存在")
            tcpCliSock.send("请求文件路径不存在".encode('utf-8'))
            tcpCliSock.close()
            continue
        path = clf.predict(path)
        path = os.path.abspath(path)
        print(path)
        tcpCliSock.send(path.encode('utf-8'))
        tcpCliSock.close()
        print('已发送')


