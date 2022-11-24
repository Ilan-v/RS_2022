import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import Config,get_data

class MatriX_trainFactorization():
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma # regularization parameter (dictionary)
        self.vec_dim = config.vec_dim
        self.epochs = config.epochs
        self.epoch = 0
        self.user_bias = None
        self.item_bias = None
        self.mu = None
        self.u = None
        self.v = None

   
    def calculate_mse(self, data: np.array):
        mse = 0
        for row in tqdm(data,desc=f"Calculating MSE"):
            user, item, rating = row
            pred_y = self.predict_on_pair(user, item)
            mse += (rating - pred_y) ** 2
        return mse / data.shape[0]

    def record(self, res_dict): #TODO: what is this?
        df = pd.DataFrame(res_dict)
        print('MF train results:')
        print(df)
        df.to_pickle('results/MF_train_results.pkl', index=False)
        # save results (optional)- for gvarim only
        # df.to_csv('results/mf_results.csv', indeX_train=False)

    def calc_regularization(self):
        return  self.gamma['item_bias'] * np.sum(self.item_bias ** 2) + \
                self.gamma['user_bias'] * np.sum(self.user_bias ** 2) + \
                self.gamma['user'] * np.sum(self.u ** 2) + \
                self.gamma['item'] * np.sum(self.v ** 2) 

    def fit(self, X_train, X_val):
        print("Fitting MF model...")
        start = time.time()
        res_dic = {'epoch': [], 'mse_train': [], 'mse_val': []}
        # read data
        n_users = np.max(X_train[:, 0]) + 1
        n_items = np.max(X_train[:, 1]) + 1
        # Initialize the model parameters
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.u = np.random.normal(0, 0.1, (self.vec_dim, n_users))
        self.v = np.random.normal(0, 0.1, (self.vec_dim, n_items))
        # get mean of rating column in data
        self.mu = np.mean(X_train[:, 2])

        # Calculating the RMSE for each epoch using SGD
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.run_epoch(X_train)
            mse_train = self.calculate_mse(X_train)
            mse_val = self.calculate_mse(X_val)
            print(f"MSE train: {round(mse_train,3)}, MSE val: {round(mse_val,3)}")
            res_dic['epoch'].append(epoch)
            res_dic['mse_train'].append(mse_train)
            res_dic['mse_val'].append(mse_val)

        self.record(res_dic)
        end = time.time()
        print(f"Fitting MF model done in {round(end - start,2)} seconds")

    def run_epoch(self, data: np.array): #TODO: what type is data
        for row in tqdm(data, desc=f"Epoch {self.epoch}"):
            user, item, rating = row
            # calculate the residual error
            pred_y = self.predict_on_pair(user, item)
            residual = rating-pred_y
            # updating the bias parameters to the opposite direction of the gradient- SGD
            self.user_bias[user] += self.lr * (residual - self.gamma['user_bias'] * self.user_bias[user])
            self.item_bias[item] += self.lr * (residual - self.gamma['item_bias'] * self.item_bias[item])
            # self.u[:, user] -= deriv_u * self.lr
            self.u[:, user] += self.lr * (residual * self.v[:, item] - self.gamma['user'] * self.u[:, user])
            self.v[:, item] += self.lr * (residual * self.u[:, user] - self.gamma['item'] * self.v[:, item])
            

    def predict_on_pair(self, user, item):
        vec_prob = np.dot(self.v[:, item], self.u[:, user])
        return self.mu + self.user_bias[user] + self.item_bias[item] + vec_prob


if __name__ == '__main__':
    config = Config(
        lr=0.01,
        gamma={'user_bias': 0.001,
                'item_bias': 0.001,
                'user': 0.001,
                'item': 0.001
            },
        vec_dim=24,
        epochs=10)
    train, validation = get_data()
    model = MatriX_trainFactorization(config)
    model.fit(train, validation)
