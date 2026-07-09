# ============================================================
# 逻辑回归 - 威斯康星乳腺癌分类案例
# ============================================================
# 任务：根据 9 个细胞特征，预测肿瘤是良性(2)还是恶性(4)
# 类型：分类（输出离散类别，不是连续数值）
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. 读取数据
# ============================================================
# CSV 共 11 列：Sample code number(编号) + 9 个特征 + Class(标签)
# 标签 Class：2=良性(benign)，4=恶性(malignant)
data = pd.read_csv(r"data\breast-cancer-wisconsin.csv", sep=",", header=0)
print(data.shape, data.ndim)  # (699, 11) 二维表：699 行样本，11 列

# ============================================================
# 2. 数据预处理
# ============================================================

# 2.0 清洗缺失值
# 原始数据里 "?" 表示缺失，需先换成 np.nan，再删除含缺失的行
new_data = data.replace("?", np.nan).dropna()
print(new_data.shape, new_data.ndim)  # (683, 11) 清洗后剩 683 条有效样本

# 2.1 分离特征 X 和标签 y
# iloc[行, 列]：按位置索引，不受列名影响
#   iloc[:, 1:-1] → 所有行，第 1 列到倒数第 2 列（跳过编号列，取 9 个特征）
#   iloc[:, -1]   → 所有行，最后一列（Class 标签）
x = new_data.iloc[:, 1:-1]
y = new_data.iloc[:, -1]
print(x.shape, x.ndim)  # (683, 9)  683 条样本，每条 9 个特征
print(y.shape, y.ndim)  # (683,)    683 个标签（一维）

# 2.2 划分训练集 / 测试集
# test_size=0.2  → 80% 训练，20% 测试
# random_state=666 → 固定随机种子，每次运行划分结果相同，便于复现
# 返回 4 个变量：
#   X_train, y_train → 用来 fit 训练（模型会看到这些数据的答案）
#   X_test,  y_test  → 用来 predict 评估（模型训练时没见过）
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=666
)

# ============================================================
# 3. 特征标准化
# ============================================================
# 逻辑回归用梯度下降，特征量纲差异大时训练慢且不稳定，需标准化
ss = StandardScaler()

# 训练集：fit_transform = ① 从 X_train 学习每个特征的均值μ和标准差σ
#                      ② 用 μ、σ 把 X_train 变换为均值0、标准差1
new_x_train = ss.fit_transform(X_train)

# 测试集：只能用 transform，用训练集学到的 μ、σ 来变换
#         不能 fit_transform，否则测试集会泄露信息，且与训练集尺度不一致
new_x_test = ss.transform(X_test)
print(new_x_train.shape, new_x_test.shape)  # 约 (546, 9)  (137, 9)

# ============================================================
# 4. 创建逻辑回归模型
# ============================================================
# 默认参数：多分类用 one-vs-rest，二分类时内部把问题转为概率再判类
lr_model = LogisticRegression()

# ============================================================
# 5. 模型训练
# ============================================================
# fit(特征, 标签)：
#   new_x_train → 标准化后的训练特征（题目）
#   y_train     → 训练标签 2 或 4（答案）
# 训练目标：学出每个特征对「属于某类」的影响权重
lr_model.fit(new_x_train, y_train)
print("=" * 50)

# ============================================================
# 6. 预测与评估
# ============================================================

# 6.1 对测试集预测（只传特征，不传标签）
y_pred = lr_model.predict(new_x_test)  # 输出每个样本的预测类别：2 或 4

print(y_pred)              # 模型预测的 137 个类别
print(y_test.tolist())     # 对应的真实标签，用于人工对比

# 6.2 方式一：手动算准确率
# accuracy = 预测对的样本数 / 总样本数
print(f"准确率:{accuracy_score(y_test, y_pred)}")

print("-" * 50)

# 6.3 查看学出的参数（了解即可）
# coef_      → 各特征的权重，正/负表示对某类概率的推高/推低
# intercept_ → 截距
print(lr_model.coef_)
print(lr_model.intercept_)

print("-" * 50)

# 6.4 方式二：模型自带的 score（底层 = predict + 算准确率）
print(f"准确率:{lr_model.score(new_x_test, y_test)}")

print("=" * 50)

# ============================================================
# 7. 分类模型常用评估指标
# ============================================================
# 这里 y_test 是真实答案，y_pred 是模型预测答案。
# 对分类模型来说，不能只看一个指标，需要结合多个角度判断模型效果。
#
# 本案例标签含义：
#   2 = 良性
#   4 = 恶性
#
# 在医学检测里，我们通常更关心「恶性 4」这一类，所以 precision / recall / F1
# 都通过 pos_label=4 指定：把 4 当作正类来计算。

# 7.1 准确率 accuracy
# 含义：所有测试样本中，模型整体预测对了多少。
# 公式：
#   accuracy = 预测正确的样本数 / 总样本数
#
# 例子：
#   137 个测试样本中预测对了 132 个，准确率就是 132 / 137。
#
# 注意：
#   准确率看整体，不关心哪一类是正类，所以不需要 pos_label。
print(f"准确率:{accuracy_score(y_test, y_pred)}")

# 7.2 精确率 precision
# 含义：模型预测为「恶性 4」的样本里面，真正是「恶性 4」的比例。
# 公式：
#   precision = TP / (TP + FP)
#
# 其中：
#   TP(True Positive)  = 真实恶性，预测恶性
#   FP(False Positive) = 真实良性，预测恶性，也叫误报
#
# 通俗理解：
#   模型说“这个人是恶性”的时候，有多大概率说对。
#
# pos_label=4：
#   告诉 sklearn：现在重点计算 4 这个类别的精确率。
print(f"精确率:{precision_score(y_test, y_pred, pos_label=4)}")

# 7.3 召回率 recall
# 含义：所有真实为「恶性 4」的样本里面，被模型成功找出来的比例。
# 公式：
#   recall = TP / (TP + FN)
#
# 其中：
#   TP(True Positive)  = 真实恶性，预测恶性
#   FN(False Negative) = 真实恶性，预测良性，也叫漏诊
#
# 通俗理解：
#   真正恶性的病人，模型能找出来多少。
#
# 在癌症检测里，召回率很重要，因为召回率低代表漏掉了较多恶性病人。
print(f"召回率:{recall_score(y_test, y_pred, pos_label=4)}")

# 7.4 F1 分数
# 含义：精确率 precision 和召回率 recall 的综合指标。
# 公式：
#   F1 = 2 * precision * recall / (precision + recall)
#
# 特点：
#   F1 不是简单平均，它会惩罚短板。
#   如果精确率很高但召回率很低，或者召回率很高但精确率很低，F1 都不会太高。
#
# 通俗理解：
#   同时看“预测恶性准不准”和“真正恶性能不能找全”。
print(f"F1分数:{f1_score(y_test, y_pred, pos_label=4)}")

# 7.5 混淆矩阵 confusion_matrix
# 含义：把每一种「真实类别 vs 预测类别」的数量列成表，方便看模型错在哪里。
#
# sklearn 默认：
#   行 = 真实值 y_test
#   列 = 预测值 y_pred
#
# 本案例标签顺序默认按从小到大排列：[2, 4]，所以矩阵结构是：
#
#              预测为 2    预测为 4
#   真实为 2      TN          FP
#   真实为 4      FN          TP
#
# 对 pos_label=4 来说：
#   TN = 真实良性，预测良性
#   FP = 真实良性，预测恶性（误报）
#   FN = 真实恶性，预测良性（漏诊）
#   TP = 真实恶性，预测恶性
#
# 通过混淆矩阵可以手动推导：
#   准确率 = (TP + TN) / 总数
#   精确率 = TP / (TP + FP)
#   召回率 = TP / (TP + FN)
print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")

# 7.6 分类报告 classification_report
# 含义：一次性输出每个类别的 precision、recall、f1-score、support。
#
# 报告里的每一行：
#   2 这一行：把良性 2 当作当前类别，计算它的精确率/召回率/F1
#   4 这一行：把恶性 4 当作当前类别，计算它的精确率/召回率/F1
#
# 列含义：
#   precision = 预测为该类别的样本中，有多少是真的
#   recall    = 真实为该类别的样本中，有多少被找出来
#   f1-score  = precision 和 recall 的综合
#   support   = 测试集中真实属于该类别的样本数量
#
# 底部几行：
#   accuracy     = 整体准确率
#   macro avg    = 各类别直接平均，不考虑每类样本数量
#   weighted avg = 按 support 加权平均，样本多的类别影响更大
print(f"分类报告:\n{classification_report(y_test, y_pred)}")
