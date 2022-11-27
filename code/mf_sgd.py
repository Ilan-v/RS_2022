import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import Config, get_data, user_item_dic_preprocess

class Matrix_Factorization_SGD():
    """
    Class for matrix factorization using SGD
    """
    def __init__(self, config):
        """
        :param config: Hyper parameters for the model
        """
        self.lr = config.lr
        self.user_items_dic = config.user_items_dic
        self.gamma = config.gamma # regularization parameter (dictionary)
        self.vec_dim = config.vec_dim
        self.epochs = config.epochs
        self.early_stop_crtiria = config.early_stop_crtiria
        self.epoch = 0
        self.user_bias = None
        self.item_bias = None
        self.mu = None
        self.u = None
        self.v = None

    def calculate_mpr(self)->float:
        """
        Calculates the mean percentage rank of the model
        Returns: The mean percentage rank score of the model
        """
        for user in self.user_items_dic:
            num_of_ranked_items = len(self.user_items_dic[user])
            ratings = [x[1] for x in self.user_items_dic[user]]
            sorted_ratings = sorted(ratings, reverse=True)
            mpr=0
            for i in range(num_of_ranked_items):
                weight=i+1/num_of_ranked_items
                mpr += weight*sorted_ratings[i]
            mpr = mpr/sum(ratings)
            return mpr

    def calculate_evaluation_measures(self, data: np.array)->dict:
        """
        Calculates the evaluation measures for the model
        Args:
            data: The data to calculate the evaluation measures on
        Returns:
            A dictionary with the evaluation measures
        """
        ssr = 0
        mae = 0
        for row in tqdm(data,desc=f"Calculating evaluation measures"):
            user, item, rating = row
            pred_y = self.predict_on_pair(user, item)
            mae+= abs(rating-pred_y)
            ssr += (rating - pred_y) ** 2
            sst= (rating - self.mu) ** 2
        r_squared = 1 - (ssr / sst)
        mse=ssr/data.shape[0]
        rmse=np.sqrt(mse)
        mae = mae/data.shape[0]
        mpr=self.calculate_mpr()
        measures={'mae':mae, 'rmse':rmse, 'r_squared':r_squared, 'mpr':mpr}
        return measures

    def calc_regularization(self): #TODO: Check if this is necessary
        return  self.gamma['item_bias'] * np.sum(self.item_bias ** 2) + \
                self.gamma['user_bias'] * np.sum(self.user_bias ** 2) + \
                self.gamma['user'] * np.sum(self.u ** 2) + \
                self.gamma['item'] * np.sum(self.v ** 2) 

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, save_path: str):
        """
        Fits the model on the training data
        Args:
            X_train: The training data
            X_val: The validation data
            save_path: The path to save the model results to
        """
        print("Fitting MF model...")
        start = time.time()
        self.X_train = X_train
        res_list = []
        rmse_list=[]
        # calculate number of users and items
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
            self.run_epoch()
            # calculate evaluation measures on train and validation data
            train_measures = self.calculate_evaluation_measures(X_train)
            val_measures = self.calculate_evaluation_measures(X_val)
            # save RMSE for early stopping criteria
            rmse_list.append(val_measures['rmse'])
            print(f"RMSE train: {round(train_measures['rmse'],3)}, RMSE val: {round(val_measures['rmse'],3)}")
            res_list.append(val_measures)

            # check for early stopping criteria
            if epoch > 0:
                if rmse_list[-2] - rmse_list[-1] < self.early_stop_crtiria:
                    print(f"Early stopping at epoch {epoch}")
                    break

        res_df = pd.DataFrame(res_list)
        res_df.to_pickle(save_path)
        end = time.time()

        print(f"Fitting MF model done in {round(end - start,2)} seconds")

    def run_epoch(self):
        """
        Runs one epoch of SGD
        """
        for row in tqdm(self.X_train, desc=f"Epoch {self.epoch}"):
            user, item, rating = row
            # calculate the residual error
            pred_y = self.predict_on_pair(user, item)
            residual = rating-pred_y
            # updating the bias parameters to the opposite direction of the gradient- SGD
            self.user_bias[user] += self.lr * (residual - self.gamma['user_bias'] * self.user_bias[user])
            self.item_bias[item] += self.lr * (residual - self.gamma['item_bias'] * self.item_bias[item])
            # updating the latent factors to the opposite direction of the gradient- SGD
            self.u[:, user] += self.lr * (residual * self.v[:, item] - self.gamma['user'] * self.u[:, user])
            self.v[:, item] += self.lr * (residual * self.u[:, user] - self.gamma['item'] * self.v[:, item])
            

    def predict_on_pair(self, user: int, item: int) -> float:
        """
        Predicts the rating of a user on an item
        Args:
            user: The user id
            item: The item id
        Returns:
            The predicted rating
        """
        vec_prob = np.dot(self.v[:, item], self.u[:, user])
        return self.mu + self.user_bias[user] + self.item_bias[item] + vec_prob

if __name__ == '__main__':
    train, validation = get_data()
    user_items_dic, item_users_dic = user_item_dic_preprocess(train)
    config = Config(
        user_items_dic=user_items_dic,
        early_stop_crtiria=0.001,
        lr=0.01,
        gamma={'user_bias': 0.001,
                'item_bias': 0.001,
                'user': 0.001,
                'item': 0.001
            },
        vec_dim=50,
        epochs=20)
    np.random.seed(42)
    model = Matrix_Factorization_SGD(config)
    model.fit(train, validation, save_path='mf_sgd.pkl')
