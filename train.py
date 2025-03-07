import argparse
import os
import torch
import torchvision.transforms as transforms
from model import VisionTransformer
from utils import CIFAR10DATASET, ImageNetDataset, create_dataloader, set_seeds, train, download_data, load_model
from config import Config

def main(config: Config, load_checkpoint: bool = False, checkpoint_path: str = None) -> None:
    """Main training function. """
        
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # train_dataset = CIFAR10DATASET(data_dir=config.dataset_path, train=True, transform=train_transforms)
    # test_dataset = CIFAR10DATASET(data_dir=config.dataset_path, train=False, transform=test_transforms)
    
    train_dataset = ImageNetDataset(data_dir=config.dataset_path, train=True, transform=train_transforms)
    test_dataset = ImageNetDataset(data_dir=config.dataset_path, train=False, transform=test_transforms)

    train_loader, test_loader, class_names = create_dataloader(train_data=train_dataset,
                                                               test_data=test_dataset,
                                                               batch_size=config.BATCH_SIZE,
                                                               num_workers=config.NUM_WORKERS,
                                                               pin_memory=config.PIN_MEMORY)
    
    vit = VisionTransformer(img_size=config.IMG_SIZE,
                            in_channels=config.CHANNELS,
                            patch_size=config.PATCH_SIZE,
                            num_encoder_layers=config.NUM_LAYERS,
                            embed_dim=config.D,
                            ff_dim=config.FF_D,
                            num_heads=config.NUM_HEADS,
                            dropout=config.DROPOUT,
                            num_classes=len(class_names))

    if load_checkpoint:
        vit = load_model(model=vit, checkpoint_path=checkpoint_path)

    optimizer = torch.optim.Adam(params=vit.parameters(),
                                 lr=config.LR,
                                 betas=config.BETAS,
                                 weight_decay=config.WEIGHT_DECAY)

    loss_fn = torch.nn.CrossEntropyLoss()

    results = train(model=vit,
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=config.EPOCHS,
                    device=config.DEVICE,
                    save_dir=config.SAVE_DIR)
    
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a ViT model.')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=Config.LR, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=Config.EPOCHS, help='Number of epochs to train')
    parser.add_argument('--dataset_path', type=str, default=Config.dataset_path, help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=Config.IMG_SIZE, help='Image size for transforms')
    args = parser.parse_args()

    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)

    set_seeds()    
    main(config)