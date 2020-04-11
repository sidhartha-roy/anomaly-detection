from config import cfg
import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio(cfg.DATASETS.RAW_PATH,
                    output=cfg.DATASETS.PATH,
                    seed=cfg.CONST.RANDOM_SEED,
                    ratio=(cfg.DATASETS.TRAIN_RATIO, cfg.DATASETS.VAL_RATIO, cfg.DATASETS.TEST_RATIO)) # default values