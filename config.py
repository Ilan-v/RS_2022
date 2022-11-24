TRAIN_PATH = "data/Train.csv"
VALIDATION_PATH = "data/Validation.csv"

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


