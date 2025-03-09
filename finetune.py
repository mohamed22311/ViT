import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import VisionTransformer
from utils import CIFAR10DATASET, create_dataloader, train
from config import Config


def freeze_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """
    Freezes all layers except the classifier head.

    Args:
        model (torch.nn.Module): Pretrained Vision Transformer model.

    Returns:
        torch.nn.Module: Model with frozen backbone.
    """
    for param in model.parameters():
        param.requires_grad = False  

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def finetune_vit(config: Config) -> None:
    """
    Fine-tunes a pretrained Vision Transformer on CIFAR-10.

    Args:
        config (Config): Configuration object containing hyperparameters.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = CIFAR10DATASET(data_dir=config.dataset_path, train=True, transform=train_transforms)
    test_dataset = CIFAR10DATASET(data_dir=config.dataset_path, train=False, transform=test_transforms)

    train_loader, test_loader, class_names = create_dataloader(
        train_data=train_dataset,
        test_data=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        in_channels=config.CHANNELS,
        patch_size=config.PATCH_SIZE,
        num_encoder_layers=config.NUM_LAYERS,
        embed_dim=config.D,
        ff_dim=config.FF_D,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        num_classes=1000,  
    )
    

    model = freeze_backbone(model)
    model.classifier = nn.Linear(config.D, len(class_names))  

    
    optimizer = torch.optim.Adam(model.classifier.parameters(),
                                 lr=config.LR,
                                 betas=config.BETAS,
                                 weight_decay=config.WEIGHT_DECAY)
    
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        save_dir=config.SAVE_DIR,
    )


if __name__ == "__main__":
    from utils import set_seeds

    set_seeds()

    config = Config()
    finetune_vit(config)
