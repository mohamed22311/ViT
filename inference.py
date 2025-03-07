import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from model import VisionTransformer
from utils import load_model
from config import Config

def preprocess_image(image_path, img_size):
    """Preprocesses the input image for the model."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model_path, config):
    """Loads the model, preprocesses the image, and makes a prediction."""
    
    model = VisionTransformer(img_size=config.IMG_SIZE,
                              in_channels=config.CHANNELS,
                              patch_size=config.PATCH_SIZE,
                              num_encoder_layers=config.NUM_LAYERS,
                              embed_dim=config.D,
                              ff_dim=config.FF_D,
                              num_heads=config.NUM_HEADS,
                              dropout=config.DROPOUT,
                              num_classes=config.NUM_CLASSES)

    model = load_model(model, model_path)
    model.eval()  

    image = preprocess_image(image_path, config.IMG_SIZE)

    device = config.DEVICE
    model.to(device)
    image = image.to(device)

    with torch.inference_mode():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    print(f"Predicted class: {config.CIFAR10_CLASS_NAMES[predicted_class]}")
    # print(f"Predicted class: {config.CIFAR10_CLASS_NAMES[predicted_class]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using ViT.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")

    args = parser.parse_args()
    
    config = Config()

    predict(args.image_path, args.model_path, config)
