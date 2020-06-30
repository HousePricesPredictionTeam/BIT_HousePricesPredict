## 使用随机森林预测房价
### 1.数据准备
* 使用的源数据集为数据预处理阶段生成的特征提取文件（numeric_feature.csv，normal_feature_gainRate.csv，SalePrice.csv), 在此基础上通过拼接合并为**TrainData.csv**文件。  
* **TrainData.csv**文件包含共计89项特征，最后一列为标签值  
* 所有输入数据均存放在data文件夹下
### 2.模型及预测结果
> 代码使用随机森林模型对房价进行预测，随机选取数据集中75%的部分为训练集,25%的部分为测试集, 输出结果为均方误差，同时生成选取的测试集标签文件 PredictionSample.csv，以及对应的房价预测结果文件 **PredictionResults.csv**