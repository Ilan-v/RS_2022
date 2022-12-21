import numpy as np
from utils import get_data, user_item_dic_preprocess
from tqdm import tqdm


class SimpleModel:
    def __init__(self, user_items_dic) -> None:
        self.user_items_dic = user_items_dic
        self.mu = 0
    
    def calculate_mpr(self)->float:
        """
        Calculates the mean percentage rank of the model
        Returns: The mean percentage rank score of the model
        """
        for user in self.user_items_dic: 
            user_avg_rating = np.mean([item[1] for item in self.user_items_dic[user]])
            num_of_ranked_items = len(self.user_items_dic[user])
            ground_truth_list = self.user_items_dic[user]
            ground_truth_dic= {}
            model_ratings_lst = []
            for item in ground_truth_list:
                item_rating = item[1]
                # convert to binary
                ground_truth_dic[item] = 0 if item_rating < user_avg_rating else 1
                # create list of model ratings for ranking
                model_ratings_lst.append((item, self.predict_on_pair(user, item[0])))

            sorted_ratings = sorted(model_ratings_lst, reverse=True)
            mpr = 0
            for i, item in enumerate(sorted_ratings):
                item_id = item[0]
                # item_ground_truth is 0 or 1
                item_ground_truth = ground_truth_dic[item_id]
                mpr += ((i+1)/num_of_ranked_items) * item_ground_truth
            mpr += mpr/sum(ground_truth_dic.values())
        
        mpr = mpr/len(self.user_items_dic)
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
        sst = 0
        for row in tqdm(data,desc=f"Calculating evaluation measures"):
            user, item, rating = row
            pred_y = self.predict_on_pair(user, item)
            mae += abs(rating-pred_y)
            ssr += (rating - pred_y) ** 2
            sst += (rating - self.mu) ** 2
        r_squared = 1 - (float(ssr) / float(sst))
        mse=ssr/data.shape[0]
        rmse=np.sqrt(mse)
        mae = mae/data.shape[0]
        mpr=self.calculate_mpr()
        measures={'mae':mae, 'rmse':rmse, 'r_squared':r_squared, 'mpr':mpr}
        return measures

    def fit(self, data: np.array) -> None:
        """
        Fits the model on the data
        Args:
            data: The data to fit the model on
        """
        self.mu = np.mean(data[:, 2])
    
    def predict_on_pair(self, user: int, item: int) -> float:
        """
        Predicts the price of the pair
        Args:
            pair: The pair to predict the price of
        Returns:
            The predicted price of the pair
        """
        return self.mu

if __name__ == '__main__':
    train, validation = get_data()
    user_items_dic, item_users_dic = user_item_dic_preprocess(train)
    model = SimpleModel(user_items_dic)
    model.fit(train)
    measures = model.calculate_evaluation_measures(validation)
    print(measures)