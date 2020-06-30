# 房价预测
使用XGBoost的回归模型对房价进行预测
1. 数据处理

   加载特征和标签，然后按0.8:0.2的比例划分训练集和测试集

   ```python
   self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(df, df2, test_size=0.2, random_state=None)
   ```

2. 训练

   训练回归模型

   ```python
   self.model = XGBRegressor( n_estimators=500, learning_rate=0.05, min_child_weight=5, max_depth=4)
   self.model.fit(self.train_X, self.train_y)
   ```

3. 测试

   在测试集上计算MSE，MAE，RMSE（取log）等指标。

   ```python
   print("Score: ", self.model.score(self.test_X, self.test_y))
   log_pre_y = np.log(self.model.predict(self.test_X))
   self.log_test_y = np.log(self.test_y)
   print("MSE: ", mean_squared_error(log_pre_y, self.test_y))
   print("MAE: ", mean_absolute_error(log_pre_y, self.test_y))
   print("LOG RMSE: ", math.sqrt(mean_squared_error(log_pre_y, self.log_test_y)))
   ```
    结果：
      ```buildoutcfg 
   Score:  0.9082311317122762
   MSE:  39636020663.29521
   MAE:  182800.04171717004
   LOG RMSE:  0.11455702056471449
      ```

