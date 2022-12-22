import pandas as pd
import numpy as np

TRAIN_PATH = "data/Train.csv"
VALIDATION_PATH = "data/Validation.csv"
TEST_PATH = "data/Test_wo_labels.csv"
TEST_RESULTS_PATH_SGD = "data/Test_SGD.csv"
TEST_RESULTS_PATH_ALS = "data/Test_ALS.csv"

USER_COL_NAME_IN_DATAEST = 'User_ID_Alias'
ITEM_COL_NAME_IN_DATASET = 'Movie_ID_Alias'
RATING_COL_NAME_IN_DATASET = 'Ratings_Rating'

# for internal use
USER_COL = 'user'
ITEM_COL = 'item'
RATING_COL = 'rating'

USERS_COL_INDEX = 0
ITEMS_COL_INDEX = 1
RATINGS_COL_INDEX = 2

BASELINE_PARAMS_FILE_PATH = 'learned_paramaters/baseline_params.pickle'
POPULARITY_DIFFERENCES_PARAMS_FILE_PATH = 'learned_paramaters/popularity_differences_params.pickle'
USER_CORRELATION_PARAMS_FILE_PATH = 'learned_paramaters/user_correlation_params.pickle'
ITEM_CORRELATION_PARAMS_FILE_PATH = 'learned_paramaters/item_correlation_params.pickle'
CSV_COLUMN_NAMES = ['item_1', 'item_2', 'sim']

def transform_data_to_internal_indexes(data: pd.DataFrame, user_map, item_map, test = False) -> pd.DataFrame:
    """
    Transforms the data to internal indexes, i.e. the indexes that are used in the code [0,num users/items].
    Args:
        data: pandas dataframe with columns [user,item,rating]
        user_map: dictionary that maps user id to internal index
        item_map: dictionary that maps item id to internal index
        test: boolean that indicates if the data is test data
    Returns:
        pandas dataframe with columns [user,item,rating] where the user and item columns are in internal indexes.
    """
    data[USER_COL] = data[USER_COL_NAME_IN_DATAEST].map(user_map)
    data[ITEM_COL] = data[ITEM_COL_NAME_IN_DATASET].map(item_map)
    if test:
        return data[[USER_COL,ITEM_COL]]
    else:
        data[RATING_COL] = data[RATING_COL_NAME_IN_DATASET]
        return data[[USER_COL,ITEM_COL, RATING_COL]]

def get_user_and_item_map(data: pd.DataFrame):
    """
    Creates a mapping from user/item id to internal index.
    Args:
        data: pandas dataframe with columns [user,item,rating]
    Returns:
        user_map: dictionary that maps user id to internal index
    """
    data[USER_COL] = pd.factorize(data[USER_COL_NAME_IN_DATAEST])[0]
    data[ITEM_COL] = pd.factorize(data[ITEM_COL_NAME_IN_DATASET])[0]
    user_map = data[[USER_COL, USER_COL_NAME_IN_DATAEST]].drop_duplicates()
    user_map = user_map.set_index(USER_COL_NAME_IN_DATAEST).to_dict()[USER_COL]
    item_map = data[[ITEM_COL, ITEM_COL_NAME_IN_DATASET]].drop_duplicates()
    item_map = item_map.set_index(ITEM_COL_NAME_IN_DATASET).to_dict()[ITEM_COL]
    return user_map, item_map

def get_data():
    """
    reads train, validation to python indices so we don't need to deal with it in each algorithm.
    of course, we 'learn' the indices (a mapping from the old indices to the new ones) only on the train set.
    
    If an index does not appear in the train set but appears in the validation set, then we can put np.nan or another indicator that tells us that.
    """
    # read train data and remap indexes 
    train = pd.read_csv(TRAIN_PATH)
    user_map, item_map = get_user_and_item_map(train)
    train = transform_data_to_internal_indexes(train, user_map, item_map)
    # read validation data and remap indexes
    validation = pd.read_csv(VALIDATION_PATH)
    validation = transform_data_to_internal_indexes(validation, user_map, item_map)
    #reads test data and remap indexes
    test = pd.read_csv(TEST_PATH)
    test = transform_data_to_internal_indexes(test, user_map, item_map, test=True)
    # drop duplicate ratings (different ratings from same user and item)
    train = train.drop_duplicates(subset=[USER_COL, ITEM_COL])
    validation = validation.drop_duplicates(subset=[USER_COL, ITEM_COL])
    test = test.drop_duplicates(subset=[USER_COL, ITEM_COL])
    # convert to numpy arrays
    train = train.values.astype(int)
    validation = validation.values.astype(int)
    test = test.values.astype(int)
    # remove rows with negative values from validation = users or items that don't exist in train set
    validation = validation[validation.min(axis=1)>=0,:]
    return train, validation, test

def user_item_dic_preprocess(data: np.array) -> dict:
    """
    Create dictionary with user as key and set of items as value.
    Args:
        data: numpy array of shape (n,3) where n is the number of ratings.
    Returns:
        dictionary with user as key and set of items as value.
    """
    items_of_user = {}
    users_of_item = {}
    for user, item, rating in data:
        if user not in items_of_user:
            items_of_user[user] = []
        if item not in users_of_item:
            users_of_item[item] = []
        # update dictionaries
        items_of_user[user].append((item, rating))
        users_of_item[item].append((user, rating))
    return items_of_user, users_of_item


class Config:
    def __init__(self, **kwargs):
        self._set_attributes(kwargs)

    def _set_attributes(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
