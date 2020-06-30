import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def load_data():
    # 导入训练集
    train_file_path = './data/TrainData.csv'
    train_data = pd.read_csv(train_file_path).drop(0)

    # 数据归一化
    train_data = normalize(train_data)

    x = train_data.iloc[:, :-1] # 前89列为特征
    y = train_data.loc[:,['Label']] # 最后一列为标签

    # 划分数据集(训练集占75%，测试集占25%)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = None)

    return train_data, x_train,x_test, y_train, y_test

# 归一化处理
def normalize(df):
    #df = (df - df.min()) / (df.max() - df.min())
    for col in df.columns[:-1]:
        df[col]= (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

# 创建随机森林模型
def model(train_x,train_y):
    my_model = RandomForestRegressor()
    my_model.fit(train_x, train_y.values.flatten())
    return my_model

def main():
    # 读取数据集并随机划分训练集和测试集
    train_data, train_x, test_x, train_y, test_y = load_data()

    # 创建随机森林模型并进行训练和预测
    my_model = model(train_x,train_y)
    predicted_prices = my_model.predict(test_x)
    #test_y_list = test_y['Label'].tolist()

    # 均方误差作为评价指标
    print("预测结果与样本的平均绝对误差为：")
    print(mean_absolute_error(test_y,predicted_prices, multioutput='uniform_average'))

    # 保存预测结果
    my_results = pd.DataFrame({'SalePrice': predicted_prices})
    my_results.to_csv('PredictionResults.csv', index=False)
    test_y.to_csv('PredictionSample.csv', index=False)

if __name__ == '__main__':
    main()




