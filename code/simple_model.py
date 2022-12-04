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