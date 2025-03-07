from torch.utils.data import Dataset
import os
import pickle
from PIL import Image
import numpy as np
from .utils import download_data


class CIFAR10DATASET(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform=None) -> None:
        """
        Custom Dataset for CIFAR-10.

        Args:
            data_dir (str): Path to the extracted CIFAR-10 dataset directory (e.g., 'cifar-10-batches-py').
            train (bool): If True, loads training data; otherwise, loads test data.
            transform (callable, optional): A function/transform to apply to the images.
        """
            
        if not os.path.exists(data_dir):
            print(f"Dataset not found at {data_dir}. Downloading...")
            download_data(source="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        destination="CIFAR10")
            
        self.data_dir = data_dir
        self.transform = transform
        
        # Get file names
        if train:
            batch_files = [i for i in os.listdir(self.data_dir) if i.startswith("data_batch")] 
        else:
            batch_files = ['test_batch']

        self.images, self.labels = self._load_batches(batch_files)

        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_batches(self, batch_files):
        """Loads data from CIFAR-10 binary files."""
        all_images, all_labels = [], []
        for batch_file in batch_files:
            batch_path = os.path.join(self.data_dir, batch_file)
            with open(batch_path, 'rb') as f:
                batch_dict = pickle.load(f, encoding='latin1') 
                images = batch_dict['data']
                labels = batch_dict['labels']

                # Reshape images (num_samples, 3, 32, 32) -> (num_samples, 32, 32, 3)
                images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                
                all_images.append(images)
                all_labels.extend(labels)
        
        return np.concatenate(all_images), np.array(all_labels)

    def load_image(self, index: int) -> Image.Image:
        """Opens an image from the dataset and returns it as a PIL Image."""
        return Image.fromarray(self.images[index])

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.images)

    def __getitem__(self, index) -> tuple:
        """Returns one sample of data: (image, label)."""
        img = self.load_image(index)
        label = self.labels[index]

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        return img, label
