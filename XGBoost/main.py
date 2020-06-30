import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import numpy as np

feature1Path = os.path.join("data", "numeric_feature.csv")
feature2Path = os.path.join("data", "normal_feature_corr.csv")
pricePath = os.path.join("data", "SalePrice.csv")

class XGModel:
    def loadData(self):
        df = pd.concat([pd.read_csv(feature1Path), pd.read_csv(feature2Path)], axis=1)

        # 归一化
        for col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        df2 = pd.read_csv(pricePath, header=None)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(df, df2, test_size=0.2, random_state=None)

    def train(self):
        self.model = XGBRegressor( n_estimators=500, learning_rate=0.05, min_child_weight=5, max_depth=4)
        self.model.fit(self.train_X, self.train_y)

    def metrics(self):
        print("Score: ", self.model.score(self.test_X, self.test_y))
        # 取log
        log_pre_y = np.log(self.model.predict(self.test_X))
        self.log_test_y = np.log(self.test_y)
        print("MSE: ", mean_squared_error(log_pre_y, self.test_y))
        print("MAE: ", mean_absolute_error(log_pre_y, self.test_y))
        print("LOG RMSE: ", math.sqrt(mean_squared_error(log_pre_y, self.log_test_y)))

def main():
    xg = XGModel()
    xg.loadData()
    xg.train()
    xg.metrics()

if __name__ == "__main__":
    main()