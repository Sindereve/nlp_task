import os

URL_DOWNLOAD_DATA = 'https://raw.githubusercontent.com/blanchefort/datasets/refs/heads/master/leroymerlin/leroymerlin.csv.zip'

# -------- DIRS ---------

DATA_DIR = 'data'
MODELS_DIR = 'data/best_models'
X_DATA_DIR = 'data/x_data'
TARGET_DIR = 'data/target'
FT_DIR = 'data/ft'

ALL_DIRS = [DATA_DIR, MODELS_DIR, X_DATA_DIR, TARGET_DIR, FT_DIR]

def _dir_searcher(name_dir: str):
    if not os.path.exists(name_dir):
        os.makedirs(name_dir)

for dir in ALL_DIRS:
    _dir_searcher(dir)
print('Структура папок проверена')


# ======== DIRS ========


# -------- OTHER ---------

RANDOM_STATE = 123

# ======== OTHER ========