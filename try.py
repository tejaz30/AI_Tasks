import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

class ResNetEmbedding:
    def __init__(self):
        self.model = resnet18(pretrained=True)  # Loading of the model ResNet-18
        self.model.fc = torch.nn.Identity()  # Replace the final layer of classification
        self.model.eval()  # Set model to evaluation mode

        # Define preprocessing steps for input images
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match ResNet input size
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
        ])

    def get_embedding(self, image):
        image_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            test_embedding = self.model(image_tensor).squeeze(0).numpy()  # Get embedding
        return test_embedding
    
if __name__ == "__main__":
    image_path = "yoboy.jpg" 
    image = Image.open(image_path).convert("RGB")  # Image loading

    extractor = ResNetEmbedding()
    test_embedding = extractor.get_embedding(image)

    print("Embedding shape:", test_embedding.shape)  # Should be (512,)
    print("First 10 values:", test_embedding[:10]) 

def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    target_embeddings={}

    for _, row in df.iterrows():
        name=row[0]
        target_embedding = np.array(row[1:], dtype = np.float32)
        target_embeddings[name] = target_embedding
    return target_embeddings


def mse(e1,e2):
    return np.mean((e1-e2)**2)

csv_path = "downloaded_file.csv"   
embeddings_dict = load_embeddings(csv_path)

employee_name = "Dave"
employee_embedding = embeddings_dict[employee_name]


mse_value= mse(test_embedding, employee_embedding)
print(mse_value)




