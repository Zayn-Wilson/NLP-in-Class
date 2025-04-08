# 贝叶斯邮件分类系统

本项目实现了一个基于朴素贝叶斯算法的邮件分类系统，可以将邮件分为垃圾邮件和普通邮件两类。系统支持两种特征提取模式：高频词模式和TF-IDF模式。

## 核心功能说明

### 1. 文本预处理
- 使用jieba分词工具进行中文分词
- 过滤无效字符和长度为1的词
- 支持自定义词典加载

### 2. 特征提取
系统支持两种特征提取模式：

#### 高频词模式
- 统计所有训练文本中出现频率最高的N个词
- 使用词频作为特征值
- 特征向量维度为N

#### TF-IDF模式
- 计算每个词的TF-IDF值
- 选择TF-IDF值最高的N个词作为特征
- 使用TF-IDF值作为特征值
- 特征向量维度为N

### 3. 分类模型
- 使用sklearn的MultinomialNB实现朴素贝叶斯分类
- 支持模型训练和预测
- 输出分类结果（垃圾邮件/普通邮件）

## 特征模式切换方法

在`classify.py`文件中，可以通过修改特征提取部分的代码来切换特征模式：

### 高频词模式
```python
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = [r'邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
```

### TF-IDF模式
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(top_num):
    """使用TF-IDF提取特征"""
    vectorizer = TfidfVectorizer(max_features=top_num)
    X = vectorizer.fit_transform([' '.join(words) for words in all_words])
    return vectorizer.get_feature_names_out()
```

切换方法：
1. 在代码中注释掉当前使用的特征提取函数
2. 取消注释要使用的特征提取函数
3. 修改相应的特征向量构建代码

## 数学公式

### 朴素贝叶斯分类器
$$
P(y|x_1, x_2, ..., x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i|y)}{P(x_1, x_2, ..., x_n)}
$$

### TF-IDF计算公式
$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)
$$

其中：
- $\text{TF}(t,d)$ 是词t在文档d中的词频
- $\text{IDF}(t) = \log\frac{N}{1 + \text{DF}(t)}$
- N是文档总数
- $\text{DF}(t)$ 是包含词t的文档数

## 选做功能

### 1. 样本平衡处理（SMOTE过采样）
系统提供了`classify_smote.py`程序，使用SMOTE算法处理样本不平衡问题。主要实现如下：

```python
# 使用SMOTE进行过采样
from imblearn.over_sampling import SMOTE

# 在训练模型前进行样本平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 使用平衡后的数据训练模型
model = MultinomialNB()
model.fit(X_resampled, y_resampled)
```

SMOTE算法通过以下步骤生成新样本：
1. 对于少数类样本$x_i$，找到其k个最近邻样本
2. 随机选择一个最近邻样本$x_{zi}$
3. 在$x_i$和$x_{zi}$的连线上随机生成新样本：
   $$x_{new} = x_i + \lambda(x_{zi} - x_i)$$
   其中$\lambda$是[0,1]之间的随机数

### 2. 模型评估指标
系统提供了`classify_evaluation.py`程序，实现全面的模型评估功能：

```python
# 计算评估指标
from sklearn.metrics import classification_report, confusion_matrix

# 输出分类报告
print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix.png')
```

评估指标计算公式：
- 精确率：$Precision = \frac{TP}{TP + FP}$
- 召回率：$Recall = \frac{TP}{TP + FN}$
- F1分数：$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

其中：
- TP：真正例（正确预测为正例）
- FP：假正例（错误预测为正例）
- FN：假负例（错误预测为负例）
