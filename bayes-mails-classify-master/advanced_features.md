# 高级功能说明

## 5. 样本平衡处理

### 问题背景
在邮件分类任务中，存在严重的样本不平衡问题：
- 垃圾邮件：127条
- 普通邮件：24条
这种不平衡会导致模型偏向于预测多数类（垃圾邮件），影响分类效果。

### SMOTE过采样实现
系统使用SMOTE（Synthetic Minority Over-sampling Technique）算法来解决样本不平衡问题：

```python
from imblearn.over_sampling import SMOTE

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### 实现原理
1. 对少数类样本进行分析
2. 在特征空间中合成新的少数类样本
3. 保持多数类样本不变
4. 最终实现两类样本数量平衡

### 注意事项
- 仅对训练集进行过采样
- 保持测试集原始分布
- 使用random_state确保结果可复现

## 6. 模型评估指标

### 1. 分类报告
系统提供详细的分类评估报告，包括：
- 精确率（Precision）：正确预测为正例的样本占所有预测为正例样本的比例
- 召回率（Recall）：正确预测为正例的样本占所有实际为正例样本的比例
- F1分数（F1-Score）：精确率和召回率的调和平均数
- 支持度（Support）：每个类别的样本数量

### 2. 混淆矩阵
系统自动生成混淆矩阵可视化：
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
```

### 3. 详细指标计算
```python
from sklearn.metrics import precision_recall_fscore_support

# 计算每个类别的详细指标
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# 输出普通邮件指标
print(f"普通邮件:")
print(f"  精确率: {precision[0]:.4f}")
print(f"  召回率: {recall[0]:.4f}")
print(f"  F1分数: {f1[0]:.4f}")

# 输出垃圾邮件指标
print(f"垃圾邮件:")
print(f"  精确率: {precision[1]:.4f}")
print(f"  召回率: {recall[1]:.4f}")
print(f"  F1分数: {f1[1]:.4f}")
```

### 评估指标计算公式

#### 精确率（Precision）
$$
Precision = \frac{TP}{TP + FP}
$$
- TP：真正例（正确预测为正例）
- FP：假正例（错误预测为正例）

#### 召回率（Recall）
$$
Recall = \frac{TP}{TP + FN}
$$
- FN：假负例（错误预测为负例）

#### F1分数（F1-Score）
$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### 使用说明
1. 运行`classify_evaluation.py`文件
2. 查看控制台输出的分类报告
3. 检查生成的混淆矩阵图（confusion_matrix.png）
4. 根据评估指标调整模型参数 