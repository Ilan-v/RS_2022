import pandas as pd
import numpy as np
import math
from tqdm import tqdm

def create_user_items_dict(df: pd.DataFrame) -> dict:
    user_items_dict = {}
    for user, item in zip(df['UserID'], df['ItemID']):
        if user not in user_items_dict:
            user_items_dict[user] = []
        user_items_dict[user].append(item)
    return user_items_dict

def sample_negative_examples_randomly(user_items_dict:dict, items_list:list)->list:
    negative_samples={}
    for user in tqdm(user_items_dict):
        relevant_samples = [x for x in items_list if x not in user_items_dict[user]]
        number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))
        negative_samples[user] = list(np.random.choice(relevant_samples, number_of_samples, replace=False))
    return negative_samples

def sample_negative_examples_by_popularity(user_items_dict:dict, items_list:list, item_probability_dict:dict)->list:
    negative_samples={}
    for user in tqdm(user_items_dict):
        relevant_samples = [x for x in items_list if x not in set(user_items_dict[user])]
        probabilities = [item_probability_dict[x] for x in relevant_samples]
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))
        negative_samples[user] = list(np.random.choice(relevant_samples, number_of_samples, replace=False, p=probabilities))
    return negative_samples

def create_item_popularity_dict(train_set:pd.DataFrame)->dict:    
    popularity_df = train_set.groupby('ItemID').size().reset_index(name='counts')
    popularity_df['probability'] = popularity_df['counts'] / popularity_df['counts'].sum()
    item_probability_dict = dict(zip(popularity_df['ItemID'], popularity_df['probability']))
    return item_probability_dict

def create_items_embeddings(items_list:list, alpha_item ,k:int)->dict:
    items_embeddings = {}
    for item in items_list:
        items_embeddings[item] = np.random.normal(0, alpha_item, k)
    return items_embeddings

def create_users_embeddings(user_items_dict:dict, alpha_user, k:int)->dict:
    users_embeddings = {}
    for user in user_items_dict:
        users_embeddings[user] = np.random.normal(0, alpha_user, k)
    return users_embeddings

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def training_loop(user_items_dict:dict,
                    items_list:list,
                    alpha_user:float,
                    alpha_item:float,
                    k:int,
                    lr:float,
                    epochs:int,
                    user_negative_samples_by_popularity:dict,
                    user_negative_samples_randomly:dict,
                    sample_negative_by_popularity:bool=False)->tuple:
    items_embeddings = create_items_embeddings(items_list, alpha_item, k)
    users_embeddings = create_users_embeddings(user_items_dict, alpha_user, k)

    for epoch in tqdm(range(epochs)):
        for user in tqdm(user_items_dict):
            if sample_negative_by_popularity:
                negative_item = user_negative_samples_by_popularity(user)
            else:
                negative_item = user_negative_samples_randomly(user)
            for item in user_items_dict[user]:
                prediction = sigmoid(np.dot(users_embeddings[user], items_embeddings[item]))
                error = 1 - prediction
                users_embeddings[user] += lr * error * items_embeddings[item] - alpha_user * users_embeddings[user]
                items_embeddings[item] += lr * error * users_embeddings[user] - alpha_item * items_embeddings[item]

            for item in negative_item:
                prediction = sigmoid(-1*(users_embeddings[user]).T.dot(items_embeddings[item]))
                error = 0 - prediction
                users_embeddings[user] += lr * error * items_embeddings[item] - alpha_user * users_embeddings[user]
                items_embeddings[item] += lr * error * users_embeddings[user] - alpha_item * items_embeddings[item]
            
    return users_embeddings, items_embeddings