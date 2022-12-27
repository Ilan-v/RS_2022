import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from utils import *

# set warnings as errors
import warnings
warnings.filterwarnings('error')

train_set_raw = pd.read_csv('data/train.csv')
# sample 3 positive items per user
df_val = train_set_raw.groupby('UserID').sample(n=3, random_state=10)
# take the rest of the data as validation set
df_train = train_set_raw[~train_set_raw.index.isin(df_val.index)].copy()
items_list = list(train_set_raw['ItemID'].unique())
users_list = list(train_set_raw['UserID'].unique())
user_items_dict_train = create_user_items_dict(df_train)
user_items_dict_val = create_user_items_dict(df_val)

# train set
train_negative_random = load_negative_samples(None, None, 'train', 'random')
train_negative_popularity = load_negative_samples(None, None, 'train', 'popularity')
# validation set
val_negative_random = load_negative_samples(None, None, 'validation', 'random')
val_negative_popularity = load_negative_samples(None, None, 'validation', 'popularity')

with open('data/train_datasets/train_random.pkl', 'rb') as f:
    df_random = pickle.load(f)
with open('data/train_datasets/train_popularity.pkl', 'rb') as f:
    df_popularity = pickle.load(f)

radnom_users_embeddings, random_items_embeddings = training_loop(
                                                    df_random,
                                                    user_items_dict_val, val_negative_random,
                                                    users_list, items_list,
                                                    alpha_item = 1e-5,
                                                    alpha_user = 1e-5,
                                                    item_init_noise=0.1,
                                                    user_init_noise=0.1,
                                                    epochs = 10,
                                                    k = 16,
                                                    lr = 0.1)