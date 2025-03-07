from torch.utils.data import Dataset
import os
from PIL import Image
from .utils import download_data

class ImageNetDataset(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform=None) -> None:
        """
        Custom Dataset for ImageNet.

        Args:
            data_dir (str): Path to the ImageNet dataset directory.
            train (bool): If True, loads training data; otherwise, loads validation data.
            transform (callable, optional): A function/transform to apply to the images.
        """

        if not os.path.exists(data_dir):
            print(f"Dataset not found at {data_dir}. Downloading...")
            download_data(source="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        destination="ImageNet")
            
        self.data_dir = os.path.join(data_dir, "train" if train else "val")
        self.transform = transform

        self.image_paths, self.labels, self.classes = self._load_image_paths_and_labels()

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def _load_image_paths_and_labels(self):
        """
        Loads image file paths and corresponding labels from the dataset directory.
        Assumes data is structured in subdirectories by class.
        """
        image_paths = []
        labels = []
        classes = sorted(os.listdir(self.data_dir))  # Class names from directory names

        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_idx)

        return image_paths, labels, classes

    def load_image(self, index: int) -> Image.Image:
        """Opens an image from the dataset and returns it as a PIL Image."""
        return Image.open(self.image_paths[index]).convert("RGB")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple:
        """Returns one sample of data: (image, label)."""
        img = self.load_image(index)
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label
