import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import pickle

def create_user_items_dict(df: pd.DataFrame) -> dict:
    """
    Creates a dictionary with user id as key and list of items as value.
    Args:
        df: dataframe with columns [userID,itemID]
    Returns:
        user_items_dict: dictionary with user id as key and list of items as value.
    """
    user_items_dict = {}
    for user, item in zip(df['UserID'], df['ItemID']):
        if user not in user_items_dict:
            user_items_dict[user] = []
        user_items_dict[user].append(item)
    return user_items_dict

def sample_negative_examples_randomly(user_items_dict:dict, items_list:list)->list:
    """
    Samples negative examples randomly. for each user, samples the same number of negative examples as the number of positive examples.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        items_list: list of all items.
    Returns:
        negative_samples: dictionary with user id as key and list of negative items as value.
    """
    negative_samples={}
    for user in tqdm(user_items_dict):
        relevant_samples = [x for x in items_list if x not in set(user_items_dict[user])]
        number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))
        negative_samples[user] = list(np.random.choice(relevant_samples, number_of_samples, replace=False))
    return negative_samples

def sample_negative_examples_by_popularity(user_items_dict:dict, items_list:list, item_probability_dict:dict)->list:
    """
    Samples negative examples by popularity. for each user, samples the same number of negative examples as the number of positive examples.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        items_list: list of all items.
        item_probability_dict: dictionary with item id as key and popularity of item as value.
    Returns:
        negative_samples: dictionary with user id as key and list of negative items as value.
    """
    negative_samples={}
    for user in tqdm(user_items_dict):
        # items that are not in the user's history
        relevant_samples = [x for x in items_list if x not in set(user_items_dict[user])]
        # calculate the probability of each item according to its popularity
        probabilities = [item_probability_dict[x] for x in relevant_samples]
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))
        negative_samples[user] = list(np.random.choice(relevant_samples, number_of_samples, replace=False, p=probabilities))
    return negative_samples

def create_item_popularity_dict(train_set:pd.DataFrame)->dict:
    """
    Creates a dictionary with item id as key and popularity of item as value.
    Args:
        train_set: dataframe with columns [userID,itemID]
    Returns:
        item_probability_dict: dictionary with item id as key and popularity of item as value.
    """
    popularity_df = train_set.groupby('ItemID').size().reset_index(name='counts')
    popularity_df['probability'] = popularity_df['counts'] / popularity_df['counts'].sum()
    item_probability_dict = dict(zip(popularity_df['ItemID'], popularity_df['probability']))
    return item_probability_dict

def load_negative_samples(  user_items_dict,
                            items_list,
                            dataset_type: str, sample_strategy: str,
                            popularity_dict=None)-> dict:
    """
    Loads data from pickle file.
    Args:
        user_items_dict: {user_id: list of positive items}
        items_list: list of all item ids
        dataset_type: 'train' or 'validation'
        sample_strategy: 'random' or 'popularity'
        popularity_dic: {item_id: popularity}
    Returns:
        dictionary: {user_id: list of items}
    """
    file_name = f'data/negative_samples/{dataset_type}_{sample_strategy}.pkl'
    try:
        with open(file_name, 'rb') as f:
            negative_samples = pickle.load(f)
    except:
        if sample_strategy == 'random':
            print(f'creating {dataset_type} negative samples randomly')
            negative_samples = sample_negative_examples_randomly(user_items_dict, items_list)
        elif sample_strategy == 'popularity':
            print(f'creating {dataset_type} negative samples by popularity')
            negative_samples = sample_negative_examples_by_popularity(user_items_dict, items_list, popularity_dict)
        with open(file_name, 'wb') as f:
            pickle.dump(negative_samples, f)

def create_items_embeddings(items_list:list, alpha_item ,k:int)->dict:
    """
    create items latent space vectors.
    Args:
        items_list: list of all items.
        alpha_item: standard deviation of the normal distribution for random init.
        k: number of latent factors.
    Returns:
        items_embeddings: dictionary with item id as key and latent space vector as value.
    """
    items_embeddings = {}
    for item in items_list:
        items_embeddings[item] = np.random.normal(0, alpha_item, k)
    return items_embeddings

def create_users_embeddings(user_items_dict:dict, alpha_user, k:int)->dict:
    """
    create users latent space vectors.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        alpha_user: standard deviation of the normal distribution for random init.
        k: number of latent factors.
    Returns:
        users_embeddings: dictionary with user id as key and latent space vector as value.
    """
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
    """
    training loop for SGD.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        items_list: list of all items.
        alpha_user: standard deviation of the normal distribution for random init.
        alpha_item: standard deviation of the normal distribution for random init.
        k: number of latent factors.
        lr: learning rate.
        epochs: number of epochs.
        user_negative_samples_by_popularity: dictionary with user id as key and list of negative items as value.
        user_negative_samples_randomly: dictionary with user id as key and list of negative items as value.
        sample_negative_by_popularity: if True, samples negative examples by popularity. if False, samples negative examples randomly.
    """
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