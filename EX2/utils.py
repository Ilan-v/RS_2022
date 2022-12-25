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
    finally:
        return negative_samples


def create_embeddings(object_lst, alpha, k):
    """
    create latent space vectors.
    Args:
        object_lst: list of all objects.
        alpha: standard deviation of the normal distribution for random init.
        k: number of latent factors.
    Returns:
        embeddings: dictionary with object id as key and latent space vector as value.
    """
    embeddings = {}
    for obj in object_lst:
        embeddings[obj] = np.random.normal(0, alpha, k)
    return embeddings


def create_dataset(user_negative_samples, dataset):
    """
    Creates a dataframe with all positive and negative examples. Used for training.
    Args:
        user_negative_samples: {user_id: list of negative items}
        dataset: dataframe with columns [userID,itemID] for positive examples
    Returns:
        dataframe with columns [userID,itemID,Rating] where Rating is 1 for positive examples and 0 for negative examples
    """
    df = dataset.copy()
    for user in tqdm(user_negative_samples.keys()):
        neg_items = user_negative_samples[user]
        df = pd.concat([df, pd.DataFrame({'UserID':user, 'ItemID':neg_items, 'Rating':0})])
    df['Rating'] = df['Rating'].fillna(1).astype(int)
    return df
    

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def training_loop(  train_df: pd.DataFrame,
                    user_items_dict_validation: dict,
                    negative_samples_validation: dict,
                    user_list: list,
                    items_list: list,
                    alpha_item: float,
                    alpha_user: float,
                    epochs: int,
                    k: int,
                    lr: float,
                    ) -> tuple:
    """
    Training loop for SGD.
    Args:
        train_df: dataframe with columns [userID,itemID,rating]
        user_items_dict_validation: {user_id: list of positive items} for validation set
        negative_samples_validation: {user_id: list of negative items} for validation set
        user_list: list of all user ids
        items_list: list of all item ids
        alpha_item: standard deviation of the normal distribution for random init of item embeddings.
        alpha_user: standard deviation of the normal distribution for random init of user embeddings.
        epochs: number of epochs.
        k: number of latent factors.
        lr: learning rate.
    Returns:
        items_embeddings: {item_id: latent space vector}
        users_embeddings: {user_id: latent space vector}
    """
    items_embeddings = create_embeddings(items_list, alpha_item, k)
    users_embeddings = create_embeddings(user_list, alpha_user, k)
    train = train_df.values
    for e in range(epochs):
        np.random.shuffle(train)
        for user, item, rating in tqdm(train, desc=f'Epoch {e+1}'):
            prediction = sigmoid(np.dot(users_embeddings[user], items_embeddings[item]))
            error = rating - prediction
            users_embeddings[user] += lr * error * items_embeddings[item] - alpha_user * users_embeddings[user]
            items_embeddings[item] += lr * error * users_embeddings[user] - alpha_item * items_embeddings[item]
    
    # TODO: add measures calculation on validation set
    return users_embeddings, items_embeddings

def MPR_calculation(positive_samples:dict, negative_samples:dict, users_embeddings:dict, items_embeddings:dict)->float:
    MPR = 0
    for user in tqdm(positive_samples.keys(), desc='MPR calculation'):
        user_mpr=0
        for item in positive_samples[user]:
            positive_score = np.dot(users_embeddings[user], items_embeddings[item])
            negative_scores = [np.dot(users_embeddings[user], items_embeddings[item]) for item in negative_samples[user]]
            #add positive score to the list of negative scores, sort the list and find the index of the positive score
            scores = np.sort(np.append(negative_scores, positive_score))
            index = np.where(scores == positive_score)[0][0]
            user_mpr+=index+1/len(scores)
        MPR+=user_mpr/len(positive_samples[user])
    MPR = MPR/len(positive_samples.keys())
    return MPR / len(positive_samples)

def Hit_Rate_at_k(positive_samples:dict, negative_samples:dict, users_embeddings:dict, items_embeddings:dict, k):
    hit_rate = 0
    for user in tqdm(positive_samples.keys()):
        user_hit_rate=0
        items_score=[]
        for item in positive_samples[user]:
            positive_score = np.dot(users_embeddings[user], items_embeddings[item])
            items_score.append((positive_score,1))
        
        negative_scores = [np.dot(users_embeddings[user], items_embeddings[item]) for item in negative_samples[user]]
        negative_scores = [(score,0) for score in negative_scores]
        items_score.extend(negative_scores)
        items_score = sorted(items_score, key=lambda x: x[0], reverse=True)
        items_score = items_score[:k]
        user_hit_rate = sum([x[1] for x in items_score])
        user_hit_rate = user_hit_rate/k
        hit_rate+=user_hit_rate
    hit_rate = hit_rate/len(positive_samples.keys())
    return hit_rate

