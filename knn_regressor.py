# 1.导包  此处导入knn回归模型
from sklearn.neighbors import KNeighborsRegressor

# 2.准备数据(此处我们模拟数据)
# 先模拟特征数据x
x_train = [[0], [1], [2], [3]]
x_test = [[4]]

# 再模拟标签数据y (房价0->70w,1->80w,2->100w,3->110w)
y_train = [70, 80, 100, 110] # 此时就是房价
# TODO 3.需求: 预测x_test中的4对应房价
# todo 3.1 创建分类模型
knn_model = KNeighborsRegressor(n_neighbors=3)

# todo 3.2 模型训练
knn_model.fit(x_train, y_train)

# todo 3.3 模型预测
y_pred = knn_model.predict(x_test)

# todo 3.4 打印预测结果
print(f"分类预测结果为:{y_pred}") # [96.66666667]