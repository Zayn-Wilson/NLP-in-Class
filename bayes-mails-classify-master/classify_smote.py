import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words

def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = [r'邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)], all_words

def build_feature_matrix(all_words, top_words):
    """构建特征矩阵"""
    vector = []
    for words in all_words:
        word_map = list(map(lambda word: words.count(word), top_words))
        vector.append(word_map)
    return np.array(vector)

def main():
    # 获取特征词和所有词
    top_words, all_words = get_top_words(100)
    
    # 构建特征矩阵
    X = build_feature_matrix(all_words, top_words)
    
    # 构建标签向量 (0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0)
    y = np.array([1]*127 + [0]*24)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 训练模型
    model = MultinomialNB()
    model.fit(X_resampled, y_resampled)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 预测新样本
    def predict(filename):
        """对未知邮件分类"""
        words = get_words(filename)
        current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))
        result = model.predict(current_vector.reshape(1, -1))
        return '垃圾邮件' if result == 1 else '普通邮件'
    
    # 测试预测
    print("\n测试样本预测结果:")
    test_files = ['邮件_files/151.txt', '邮件_files/152.txt', '邮件_files/153.txt', 
                 '邮件_files/154.txt', '邮件_files/155.txt']
    for file in test_files:
        print(f'{file}分类情况: {predict(file)}')

if __name__ == "__main__":
    main() 