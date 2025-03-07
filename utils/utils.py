import os
from pathlib import Path
import tarfile
import zipfile
import requests
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source=link to the dataset download,
                      destination="dataset")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download the data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        if target_file.endswith(".zip"):
            # Unzip the data
            with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
                print(f"[INFO] Unzipping {target_file} data...") 
                zip_ref.extractall(image_path)
        
        elif target_file.endswith(".tar"):
            with tarfile.open(data_path / target_file, "r") as tar:
                tar.extractall(image_path)

        elif target_file.endswith(".gz"):
            with tarfile.open(data_path / target_file, "r:gz") as tar:
                tar.extractall(image_path)
        
        else:
            raise TypeError("Invalid type provided!")

        # Remove file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

def create_dataloader(train_data: Dataset,
                      test_data: Dataset,
                      batch_size: int,
                      num_workers: int,
                      pin_memory:bool):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        pin_memory: pin memory for faster data loading
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """
    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
    
    return train_dataloader, test_dataloader, class_names

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="model_2.pth")
    """

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth' "
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")

    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """Loads a PyTorch model from a given file path.

    Args:
        model: A PyTorch model instance (should be the same architecture as the saved model).
        model_path: Path to the saved model file (".pth" or ".pt").

    Returns:
        The PyTorch model with loaded weights.

    Example usage:
        model = YourModelClass()  # Initialize the model architecture
        model = load_model(model, "models/model_2.pth")
    """
    model_path = Path(model_path)
    
    assert model_path.exists(), f"Model file not found: {model_path}"
    
    print(f"[INFO] Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    
    return model

import matplotlib.pyplot as plt

def plot_loss_curves(results:dict, save_dir:str):
    """Plots and saves training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        save_dir (str): file path to save the figure (default: "loss_curves.png").
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Save figure
    save_path = save_dir + 'loss_curves.png'
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
