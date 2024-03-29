import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import Config,get_data, user_item_dic_preprocess, TEST_PATH, TEST_RESULTS_PATH_ALS
from mf_sgd import Matrix_Factorization_SGD

class Matrix_Factorization_ALS(Matrix_Factorization_SGD):
    """
    Class for matrix factorization using alternating least squares
    """
    def __init__(self, config):
        super().__init__(config)
        self.item_users_dic = config.item_users_dic
    
    # we are only updating run epoch method, so we can use the rest of the code from mf_sgd.py (fit method, predict_on_pair method, etc.)
    def run_epoch(self): 
        """
        Runs one epoch of the model, optimzing the user and item vectors.
        """
        # optimize user vectors
        for user in tqdm(self.user_items_dic.keys(), desc=f"Updating user vectors"):
            user_items = self.user_items_dic[user]
            item_mtx = np.zeros((self.vec_dim, self.vec_dim))
            residual_vector = np.zeros(self.vec_dim)
            user_bias_residual = 0
            # run over all items rated by user
            for item in user_items:
                item_index = item[0]
                item_rating = item[1]
                item_mtx += np.outer(self.v[:,item_index], self.v[:,item_index]) + self.gamma['user'] * np.eye(self.vec_dim)
                residual_vector += (item_rating - self.mu - self.item_bias[item_index] - self.user_bias[user]) * self.v[:,item_index]
                # calculations for bias 
                user_bias_residual += (item_rating - self.mu - self.item_bias[item_index]-self.u[:,user].T.dot(self.v[:,item_index]))
        # update user vector
        self.u[:,user] = np.linalg.inv(item_mtx).dot(residual_vector)
        self.user_bias[user] = user_bias_residual / (len(user_items) + self.gamma['user_bias'])

        # optimize item vectors
        for item in tqdm(self.item_users_dic, desc=f"Updating item vectors"):
            item_users = self.item_users_dic[item]
            user_mtx = np.zeros((self.vec_dim, self.vec_dim))
            residual_vector = np.zeros(self.vec_dim)
            item_bias_residual = 0
            for user in item_users:
                user_index = user[0]
                user_rating = user[1]
                user_mtx += np.outer(self.u[:,user_index], self.u[:,user_index]) + self.gamma['item'] * np.eye(self.vec_dim)
                residual_vector += (user_rating - self.mu - self.item_bias[item] - self.user_bias[user_index]) * self.u[:,user_index]
                # calculations for bias
                item_bias_residual += (user_rating - self.mu - self.user_bias[user_index] - self.u[:,user_index].T.dot(self.v[:,item]))
            # update item vector
            self.v[:,item] = np.linalg.inv(user_mtx).dot(residual_vector)
            self.item_bias[item] = item_bias_residual / (len(item_users) + self.gamma['item_bias'])

if __name__ == '__main__':
    train, validation, test = get_data()
    user_items_dic, item_users_dic = user_item_dic_preprocess(train)
    config = Config(
        early_stop_crtiria=0.001,
        gamma={'user_bias': 0.001,
                'item_bias': 0.001,
                'user': 0.001,
                'item': 0.001
            },
        vec_dim=64,
        epochs=20, 
        user_items_dic=user_items_dic,
        lr=0.001,
        item_users_dic=item_users_dic)
    np.random.seed(10)
    model = Matrix_Factorization_ALS(config)
    model.fit(train, validation, 'results/mf_als.pkl')
    # Save test results
    res_list = []
    for row in test:
        if np.any(row < 0):
            res_list.append(model.mu)
        else:    
            res_list.append(model.predict_on_pair(row[0], row[1]))
    # load the test indexes
    test = pd.read_csv(TEST_PATH)
    # create a dataframe with the results
    test['rating'] = res_list
    # save the results
    test.to_csv(TEST_RESULTS_PATH_ALS, index=False)
