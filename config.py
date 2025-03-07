import os 
import torch

class Config:
    IMG_SIZE = 224 # height and width of images 
    CHANNELS = 3 # Number of channels in images 
    BATCH_SIZE = 32 # Number of samples in each Batach
    PATCH_SIZE = 16 # height and width of image patches 
    FF_D = 3072 # feed forward dimension  
    D = 768 # embedding dimension 
    NUM_HEADS = 12 # number of encoder heads 
    NUM_LAYERS = 12 # number of encoder layers
    DROPOUT = 0.1 # dropout ratio 
    EPOCHS = 10 # number of training epochs
    LR = 1e-3 # learning rate
    BETAS = (0.9, 0.999) # betas for Adam optimizer
    NUM_WORKERS = os.cpu_count() 
    PIN_MEMORY = True # pin memory for faster data loading
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # device to compute on
    SEED = 42 # seed for reproducibility
    SAVE_DIR = "models/" # directory to save models
    # dataset_path = './data/CIFAR10'
    dataset_path = './data/imagenet'
    CIFAR10_CLASS_NAMES = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    IMAGE_NET_CLASS_NAMES = []
    