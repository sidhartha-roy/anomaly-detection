from easydict import EasyDict as edict
import torch
import torch.nn as nn


__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
__C.DATASETS                               = edict()
__C.DATASETS.PATH                          = './datasets/'
__C.DATASETS.RAW_PATH                      = './preprocessing/data/'
__C.DATASETS.TRAIN_RATIO                   = 0.8
__C.DATASETS.VAL_RATIO                     = 0.1
__C.DATASETS.TEST_RATIO                    = 0.1
__C.DATASETS.BATCH_SIZE                    = 100 # make batch size smaller if CUDA goes out of memory
__C.DATASETS.WEIGHTED_SAMPLER              = True

# constants
__C.CONST                                  = edict()
__C.CONST.RANDOM_SEED                      = 25
__C.CONST.TRAIN                            = 'train'
__C.CONST.VAL                              = 'val'
__C.CONST.TEST                             = 'test'
__C.CONST.USE_GPU                          = torch.cuda.is_available()
__C.CONST.DEVICE                           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
__C.CONST.NUM_WORKERS                      = 1
__C.CONST.PIN_MEMORY                       = True

# Dataset transformation parameters
__C.TRANSFORM                              = edict()
__C.TRANSFORM.RESIZE                       = 224
__C.TRANSFORM.MEAN                         = [0.485, 0.456, 0.406]
__C.TRANSFORM.STD                          = [0.229, 0.224, 0.225]
__C.TRANSFORM.DEGREES                      = (0, 180) # tuple with min and max angle
__C.TRANSFORM.BRIGHTNESS                   = 0 # + to - values
__C.TRANSFORM.CONTRAST                     = 0.2 # + to - values
__C.TRANSFORM.SATURATION                   = 0.2 # + to - values
__C.TRANSFORM.HUE                          = [-0.5, 0.5] # min to max hue values

# Dataset transformation parameters
__C.MODEL                                  = edict()
__C.MODEL.DIR                              = './pretrained'
__C.MODEL.FILENAME                         = './pretrained/anomaly.pt'
__C.MODEL.STATE_DICT_FILE                  = './pretrained/state_dict.pt'
__C.MODEL.CRITERION                        = nn.NLLLoss()
__C.MODEL.OPTIMIZER                        = 'ADAM'  # 'ADAM' or 'SGD'
__C.MODEL.LR                               = 0.001
__C.MODEL.MAX_EPOCHS_STOP                  = 10 # currently keep MAX_EPOCHS same as N_EPOCHS
__C.MODEL.N_EPOCHS                         = 10
__C.MODEL.TRAIN_PRINT_EVERY                = 1
__C.MODEL.HISTORY_PATH                     = './pretrained/history.pkl'
__C.MODEL.CONTINUE_TRAINING                = True
