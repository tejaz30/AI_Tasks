import pandas as pd  # Importing Pandas for handling CSV files
import numpy as np  # Importing NumPy for numerical operations
import torch  # Importing PyTorch to run the operations such as backpropogation
import torchvision.transforms as transforms  # Import transformations for image preprocessing
from torchvision.models import resnet18  # Import ResNet18 model from torchvision
from PIL import Image  # Import PIL for image handling
import torch.nn.functional as F  # Import functional API for loss functions

# Class to extract embeddings from ResNet18
class ResNetEmbedding:
    def __init__(self):
        # Load the pre-trained ResNet18 model
        self.model = resnet18(pretrained=True)
        
        # Remove the last fully connected layer since we only need embeddings
        self.model.fc = torch.nn.Identity()
        
        # Set model to evaluation mode (no training, just inference)
        self.model.eval()

        # Define the preprocessing steps required by ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  #(ResNet input size)
            transforms.ToTensor(),  # Convert image to PyTorch tensor(Someting of a datatype which represents images as matrices)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize (standard procedure for ResNet)
        ])

    def get_embedding(self, image_tensor):
        """Passes image through ResNet model to get its feature embedding"""
        return self.model(image_tensor).squeeze(0)  # Removes extra dimension (batch size of 1)

# Function to load target embeddings from a CSV file
def load_embeddings(csv_path):
    df = pd.read_csv(csv_path)  # using a Pandas DataFrame to store and process the CSV file
    target_embeddings = {}  # Dictionary to store embeddings

    # Iterate over each row in the CSV file
    for _, row in df.iterrows():
        name = row[0]  # First column is the employee's name
        target_embedding = np.array(row[1:], dtype=np.float32)  # Convert the rest of the row into a NumPy array
        target_embeddings[name] = torch.tensor(target_embedding, dtype=torch.float32)  # Convert to PyTorch tensor

    return target_embeddings  # Return dictionary with names and embeddings

# Function to calculate Mean Squared Error (MSE) between two embeddings
def mse(e1, e2):
    """Computes MSE loss to measure similarity between embeddings"""
    return torch.mean((e1 - e2) ** 2)  # Standard MSE formula

# Function to modify an image until its embedding is close to the target embedding
def generate_adversarial_image(image_path, target_embedding, extractor, lr=0.01, mse_threshold=1e-3):
    """Modifies an image until its embedding closely matches the target embedding (MSE < threshold)"""

    # Load the original image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")
    
    # Apply preprocessing and add batch dimension (1, C, H, W)
    image_tensor = extractor.preprocess(image).unsqueeze(0)

    # Convert image tensor into a trainable variable (so we can modify it)
    image_tensor = torch.nn.Parameter(image_tensor.clone(), requires_grad=True)

    # Use Adam optimizer to update the image pixel values
    optimizer = torch.optim.Adam([image_tensor], lr=lr)

    iteration = 0  # Counter to keep track of iterations
    while True:  # Keeps modifying the image until we reach the required MSE threshold
        optimizer.zero_grad()  # Clear previous gradients
        embedding = extractor.get_embedding(image_tensor)  # Get embedding for the current image
        loss = mse(embedding, target_embedding)  # Compute MSE loss (how different the embeddings are)
        loss.backward()  # Compute gradients for backpropagation
        optimizer.step()  # Update the image to reduce the loss

        iteration += 1  # Increment iteration count
        print(f"Iteration {iteration}: MSE = {loss.item():.6f}")  # Print current MSE loss

        # Stop when the loss (MSE) is below the threshold
        if loss.item() < mse_threshold:
            print(f"Image modified successfully in {iteration} iterations with MSE: {loss.item():.6f}")
            break  # Exit the loop

    # Convert modified tensor back to an image format
    modified_image = image_tensor.detach().squeeze(0).permute(1, 2, 0).numpy()  # Convert to NumPy array
    modified_image = (modified_image * 255).clip(0, 255).astype(np.uint8)  # Scale values to 0-255 and clip invalid values
    return Image.fromarray(modified_image)  # Convert back to PIL image

# The execution of the abpve defined function to load the csv file and the generation of adverarial image starts here
if __name__ == "__main__":
    image_path = "yoboy.jpg"  # Path to the input image
    csv_path = "downloaded_file.csv"  # Path to the CSV file containing target embeddings
    employee_name = "Dave"  # Name of the employee whose embedding we want to match

    extractor = ResNetEmbedding()  # Create embedding extractor
    embeddings_dict = load_embeddings(csv_path)  # Load all target embeddings
    target_embedding = embeddings_dict[employee_name]  # Retrieve the target embedding for the selected employee

    # Generate an adversarial image that matches the target embedding
    adversarial_image = generate_adversarial_image(image_path, target_embedding, extractor)

    # Save the modified image
    adversarial_image.save("adversarial_yoboy.jpg")

    print("Modified image saved as adversarial_yoboy.jpg")
