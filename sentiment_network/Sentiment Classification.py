# 定义如何显示评论
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")


# What we know!获取评论数据
g = open('reviews.txt', 'r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()
# What we WANT to know!获取评价数据
g = open('labels.txt', 'r')
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

from collections import Counter
import numpy as np
import time
import sys

'''
# 定义词频计数器
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

# 统计不同结果词频
for i in range(len(reviews)):
    if (labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1

# 定义好坏比例统计数器
pos_neg_ratios = Counter()

# 计算不同结果词频比率
for term, cnt in list(total_counts.most_common()):
    if (cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
        pos_neg_ratios[term] = pos_neg_ratio

# 标准化比率正态化
for word, ratio in pos_neg_ratios.most_common():
    if (ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

vocab = set(total_counts.keys())
vocab_size = len(vocab)

# 定义输入标准化的字典
layer_0 = np.zeros((1, vocab_size))

# 定义词的序列字典
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


# 输入文字根据word2index转词频
def update_input_layer(review):
    global layer_0
    # clear out previous state, reset the layer to be all 0s
    layer_0 *= 0
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1


# 定义输出数据转数字
def get_target_for_label(label):
    if (label == 'POSITIVE'):
        return 1
    else:
        return 0
'''

# 定义神经网络
class SentimentNetwork:
    def __init__(self, reviews, labels, min_count=10, polarity_cutoff=0.1, hidden_nodes=10, learning_rate=0.1):

        # set our random number generator
        np.random.seed(1)
        ##project6初始化筛选特征
        self.pre_process_data(reviews, polarity_cutoff, min_count)

        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    # 根据评论和标签初始化词的字典和标签字典
    def pre_process_data(self, reviews, polarity_cutoff, min_count):
        # project6计算不同结果的词频比率
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if (labels[i] == 'POSITIVE'):
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term, cnt in list(total_counts.most_common()):
            if (cnt >= 50):
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word, ratio in pos_neg_ratios.most_common():
            if (ratio > 1):
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))

        '''
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        self.review_vocab = list(review_vocab)
        '''
        # project6筛选特征
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if (total_counts[word] > min_count):  # 如果词频大于min_count
                    if (word in pos_neg_ratios.keys()):  # 并且词的的正态化比率大于polarity_cutoff
                        if ((pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff)):
                            review_vocab.add(word)  # 加入特征
                    else:
                        review_vocab.add(word)
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    # 初始化网络超参数
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        self.weights_1_2 = np.random.normal(0.0, self.output_nodes ** -0.5,
                                            (self.hidden_nodes, self.output_nodes))

        self.learning_rate = learning_rate

        self.layer_0 = np.zeros((1, input_nodes))
        # project5 增加layer_1
        self.layer_1 = np.zeros((1, hidden_nodes))

    # 评论转词频矩阵
    def update_input_layer(self, review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
        for word in review.split(" "):
            if (word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1  # project4减少噪声权重，不统计词频

    # 标签转01输出
    def get_target_for_label(self, label):
        if (label == 'POSITIVE'):
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        # project5减少噪声权重，统计每个评论中那些词出现过
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if (word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        assert (len(training_reviews) == len(training_labels))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]

            #### Implement the forward pass here ####
            ### Forward pass ###

            # Input Layer
            # project5输入修改
            # self.update_input_layer(review)

            # Hidden layer
            # layer_1 = self.layer_0.dot(self.weights_0_1)
            # project5减少噪声权重，统计每个评论中那些词出现过
            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]

            # Output layer
            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            #### Implement the backward pass here ####
            ### Backward pass ###

            layer_2_error = layer_2 - self.get_target_for_label(
                label)  # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)  # errors propagated to the hidden layer
            layer_1_delta = layer_1_error  # hidden layer gradients - no nonlinearity so it's the same as the error

            self.weights_1_2 -= self.layer_1.T.dot(
                layer_2_delta) * self.learning_rate  # update hidden-to-output weights with gradient descent step
            '''
            self.weights_0_1 -= self.layer_0.T.dot(
                layer_1_delta) * self.learning_rate  # update input-to-hidden weights with gradient descent step
            '''
            for index in review:
                self.weights_0_1[index] -= layer_1_delta[
                                               0] * self.learning_rate  # update input-to-hidden weights with gradient descent step

            if (np.abs(layer_2_error) < 0.5):
                correct_so_far += 1

            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write(
                "\rProgress:" + str(100 * i / float(len(training_reviews)))[:4] + "% Speed(reviews/sec):" + str(
                    reviews_per_second)[0:5] + " #Correct:" + str(correct_so_far) + " #Trained:" + str(
                    i + 1) + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")

    def test(self, testing_reviews, testing_labels):

        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            reviews_per_second = i / float(time.time() - start)

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + "% #Correct:" + str(correct) + " #Tested:" + str(i + 1) + " Testing Accuracy:" + str(
                correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):

        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

        if (layer_2[0] > 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"


mlp = SentimentNetwork(reviews[:-1000], labels[:-1000], min_count=20, polarity_cutoff=0.5, learning_rate=0.0001)
mlp.train(reviews[:-3000], labels[:-3000])
print('')
mlp.test(reviews[-1000:], labels[-1000:])
