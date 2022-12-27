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


def sample_negative_examples_randomly(user_items_dict:dict,
                                      items_list:list,
                                      dataset_type: str)->list:
    """
    Samples negative examples randomly. for each user, samples the same number of negative examples as the number of positive examples.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        items_list: list of all items.
        dataset_type: 'train' or 'validation'
    Returns:
        negative_samples: dictionary with user id as key and list of negative items as value.
    """
    negative_samples={}
    for user in tqdm(user_items_dict):
        relevant_samples = [x for x in items_list if x not in set(user_items_dict[user])]
        if dataset_type == 'validation':
            number_of_samples = 97
        elif dataset_type == 'train':
            number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))

        negative_samples[user] = list(np.random.choice(relevant_samples, number_of_samples, replace=False))
    return negative_samples


def sample_negative_examples_by_popularity( user_items_dict:dict,
                                            items_list:list,
                                            item_probability_dict:dict,
                                            dataset_type: str)->list:
    """
    Samples negative examples by popularity. for each user, samples the same number of negative examples as the number of positive examples.
    Args:
        user_items_dict: dictionary with user id as key and list of items as value.
        items_list: list of all items.
        item_probability_dict: dictionary with item id as key and popularity of item as value.
        dataset_type: 'train' or 'validation'
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
        # number of negative samples to sample
        if dataset_type == 'validation':
            number_of_samples = 97
        elif dataset_type == 'train':
            number_of_samples = min(len(relevant_samples), len(user_items_dict[user]))
        # sample negative examples
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
            negative_samples = sample_negative_examples_randomly(user_items_dict, items_list,dataset_type=dataset_type)
        elif sample_strategy == 'popularity':
            print(f'creating {dataset_type} negative samples by popularity')
            negative_samples = sample_negative_examples_by_popularity(user_items_dict, items_list, popularity_dict, dataset_type=dataset_type)
        with open(file_name, 'wb') as f:
            pickle.dump(negative_samples, f)
    finally:
        return negative_samples


def create_embeddings(object_lst, std, k):
    """
    create latent space vectors.
    Args:
        object_lst: list of all objects.
        std: standard deviation of the normal distribution for random init.
        k: number of latent factors.
    Returns:
        embeddings: dictionary with object id as key and latent space vector as value.
    """
    embeddings = {}
    for obj in object_lst:
        embeddings[obj] = np.random.normal(0, std, k)
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
    eps = 1e-8
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1-eps if x > 0 else eps


def validation_regularization(user_embs, item_embs, alpha_item, alpha_user):
    """
    Calculate regularization loss for user and item embeddings.
    Args:
        user_embs (dict): dictionary of user embeddings
        item_embs (dict): dictionary of item embeddings
        reg_lambda (float): regularization parameter
    """
    user_reg = 0
    item_reg = 0
    for user in user_embs:
        user_reg += np.linalg.norm(user_embs[user])
    for item in item_embs:
        item_reg +=  np.linalg.norm(item_embs[item])
    return ((alpha_user/2) * user_reg)/len(user_embs) + ((alpha_item/2) * item_reg/len(item_embs))

def validation_log_loss(positive_samples: dict,
                        negative_samples: dict,
                        user_embeddings: dict,
                        item_embeddings: dict,
                        )-> float:
    """
    Calculate log loss for a given set of positive and negative samples per user.
    Args:
        positive_samples (dict): dictionary of positive samples per user
        negative_samples (dict): dictionary of negative samples per user
        user_vectors (dict): dictionary of user vectors
        item_vectors (dict): dictionary of item vectors
    """
    loss = 0
    for user in tqdm(positive_samples):
        # get user vector
        user_vector = user_embeddings[user]
        # get positive and negative items vectors 
        pos_item_vectors = [item_embeddings[x] for x in positive_samples[user]]
        neg_item_vectors = [item_embeddings[x] for x in negative_samples[user]]
        # convert lists of arrays to matrices
        pos_items_matrix = np.vstack(pos_item_vectors)
        neg_items_matrix = np.vstack(neg_item_vectors)
        # use epsilon to avoid log(0)
        eps = 1e-8
        # calculate loss for positive items
        pos_loss = np.log(eps + np.array([sigmoid(x) for x in np.dot(user_vector, pos_items_matrix.T)]))
        # calculate loss for negative item
        neg_loss = np.log(1 + eps - np.array([sigmoid(x) for x in np.dot(user_vector, neg_items_matrix.T)]))
        # add up losses
        loss += np.sum(pos_loss)/len(pos_loss) + np.sum(neg_loss)/len(neg_loss)
    
    log_loss = -loss/(len(positive_samples))
    return log_loss 

def validation_loss_func(   positive_samples,
                            negative_samples,
                            user_embeddings,
                            item_embeddings,
                            alpha_user,
                            alpha_item) -> float:
    log_loss = validation_log_loss(positive_samples, negative_samples, user_embeddings, item_embeddings)
    reg = validation_regularization(user_embeddings, item_embeddings, alpha_user, alpha_item)
    return log_loss + reg

def train_loss_func(prediction: float,
                    rating: int,
                    user_embedding: np.ndarray,
                    item_embedding: np.ndarray,
                    alpha_item: float,
                    alpha_user: float
                    ) -> float:
    """
    Calculate loss for a given prediction, rating, user embedding and item embedding.
    Args:
        prediction (float): prediction for a given user-item pair
        rating (int): actual rating for a given user-item pair
        user_embedding (np.ndarray): user embedding
        item_embedding (np.ndarray): item embedding
    """
    if rating == 1:
        log_loss = -np.log(sigmoid(prediction))
    else:
        log_loss = -np.log(1 - sigmoid(prediction))
    regularization = (alpha_user/2) * np.linalg.norm(user_embedding) + (alpha_item/2) * np.linalg.norm(item_embedding)
    return log_loss + regularization


def training_loop(  train_df: pd.DataFrame,
                    user_items_dict_validation: dict,
                    negative_samples_validation: dict,
                    user_list: list,
                    items_list: list,
                    alpha_item: float,
                    alpha_user: float,
                    item_init_noise: float,
                    user_init_noise: float,
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
    items_embeddings = create_embeddings(items_list, item_init_noise, k)
    users_embeddings = create_embeddings(user_list, user_init_noise, k)
    train = train_df.values
    # loss objects
    loss_increase_counter = 0
    loss_dic = {'train':[], 'validation':[]}
    for e in range(epochs):

        np.random.shuffle(train)
        loss = 0
        for user, item, rating in tqdm(train, desc=f'Epoch {e+1}'):
            prediction = sigmoid(np.dot(users_embeddings[user], items_embeddings[item]))
            error = rating - prediction
            users_embeddings[user] += lr * (error * items_embeddings[item] - alpha_user * users_embeddings[user])
            items_embeddings[item] += lr * (error * users_embeddings[user] - alpha_item * items_embeddings[item])
            # calculate loss
            loss += train_loss_func(prediction, rating, users_embeddings[user], items_embeddings[item], alpha_item, alpha_user)

        # calculate validation loss
        val_loss = validation_loss_func(positive_samples=user_items_dict_validation,
                                        negative_samples=negative_samples_validation,
                                        user_embeddings=users_embeddings,
                                        item_embeddings=items_embeddings,
                                        alpha_item=alpha_item,
                                        alpha_user=alpha_user)
        epoch_train_loss = loss/len(train)
        loss_dic['train'].append(epoch_train_loss)
        loss_dic['validation'].append(val_loss)
        print(f'Epoch {e+1} train loss: {round(epoch_train_loss,3)} validation loss: {round(val_loss,3)}')

        # check for early stopping
        if e!=0:
            if loss_dic['validation'][-1] > loss_dic['validation'][-2]:
                loss_increase_counter += 1
                if loss_increase_counter == 2:
                    print('Early stopping')
                    # break
            else: 
                loss_increase_counter = 0
    return users_embeddings, items_embeddings

def MPR_calculation(positive_samples:dict, negative_samples:dict, users_embeddings:dict, items_embeddings:dict)->float:
    MPR = 0
    for user in tqdm(positive_samples.keys(), desc='MPR calculation'):
        user_mpr=0
        for item in positive_samples[user]:
            positive_score = np.dot(users_embeddings[user], items_embeddings[item])
            negative_scores = [np.dot(users_embeddings[user], items_embeddings[item]) for item in negative_samples[user]]
            neg_lst = [(x,0) for x in negative_scores]
            scores =  neg_lst + [(positive_score,1)]
            #add positive score to the list of negative scores, sort the list and find the index of the positive score
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            for i in range(len(scores)):
                rating = scores[i][1]
                user_mpr += rating*(i+1)/len(scores)

        MPR+=user_mpr/len(positive_samples[user])
    MPR = MPR/len(positive_samples.keys())
    return MPR 

def Hit_Rate_at_k(positive_samples:dict, negative_samples:dict, users_embeddings:dict, items_embeddings:dict, k):
    """
    Calculate average hitrate@k for a given set of positive and negative samples per user.
    Args:
        positive_samples (dict): dictionary of positive samples per user
        negative_samples (dict): dictionary of negative samples per user
        users_embeddings (dict): dictionary of user embeddings
        items_embeddings (dict): dictionary of item embeddings
        k (int): number of items to consider for hitrate calculation
    """
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
        user_hit_rate = user_hit_rate/len(positive_samples[user])
        hit_rate+=user_hit_rate
    hit_rate = hit_rate/len(positive_samples.keys())
    return hit_rate