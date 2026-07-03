# ============================================================
# 波士顿房价数据集 - 手动加载方式
# ============================================================
# 背景：sklearn 1.2+ 已移除 load_boston()，运行旧代码会报 ImportError，
#       报错信息里会给出下面这套「从原始网站读取」的替代代码。
# ============================================================

# 【已废弃】旧版 sklearn 一行导入（新版会报错，故注释掉）
# from sklearn.datasets import load_boston
# boston = load_boston()
# X = boston.data      # 特征矩阵 (506, 13)
# y = boston.target    # 房价标签 (506,)

import pandas as pd   # 读取远程 CSV 文本
import numpy as np    # 数组拼接与切片
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TODO 1.准备数据
# todo 1.1 获取原始数据
# 波士顿房价数据的官方原始地址（卡内基梅隆大学统计系）
data_url = "http://lib.stat.cmu.edu/datasets/boston"

# 从 URL 读取原始文本，解析成 DataFrame
# - sep=r"\s+"  : 按「一个或多个空白字符」分列（空格/制表符等）
#                 前面的 r 表示原始字符串，避免 \s 被 Python 当成非法转义
# - skiprows=22 : 跳过文件前 22 行说明文字，从实际数据行开始读
# - header=None : 文件没有列名表头，全部按数字读入
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

print("=" * 50)
print(raw_df.head())

# ---------- 原始文件的特殊格式（重点） ----------
# CMU 网站上的 boston 文件不是「一行一样本」，而是特征和标签交错存放：
#
#   第 0 行（偶数行）: 11 个特征值
#   第 1 行（奇数行）: 11 列，其中前 2 列是剩余特征，第 3 列（索引 2）是房价标签
#   第 2 行（偶数行）: 11 个特征值
#   第 3 行（奇数行）: 同上
#   ... 共 506 组，即 1012 行
#
# 13 个特征名（了解即可）：
#   CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
# 标签 MEDV：该区域房价中位数，单位是「千美元」

# [::2, :]    → 从第 0 行开始，步长 2，取所有偶数行的全部 11 列
# [1::2, :2]  → 从第 1 行开始，步长 2，取所有奇数行的前 2 列
# np.hstack   → 左右横向拼接 → 11 + 2 = 13 列，即完整特征矩阵
# 最终形状：(506, 13)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

# [1::2, 2] → 所有奇数行的第 3 列（索引 2），即房价标签 MEDV
# 最终形状：(506,)
target = raw_df.values[1::2, 2]

# ---------- 查看结果 ----------
print(f"特征矩阵形状: {data.shape}")       # 预期 (506, 13)
print(f"标签数组形状: {target.shape}")     # 预期 (506,)
print(f"第1条样本的13个特征: {data[0]}")
print(f"第1条样本的房价: {target[0]}")
print(f"全部房价标签: {target}")

print("=" * 50)

# todo 1.2 数据切割
X_train, X_test , Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=42)
# print("=" * 50)

# todo 1.3 特征的标准化数据
ss = StandardScaler()
new_X_train =  ss.fit(X_train)
new_X_test = ss.transform(X_test) # 测试集只能用transform()转换,因为前面训练集已经训练了模型计算了相关内容