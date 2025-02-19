**Adversarial Image Generation using ResNet Embeddings**

This project is built for adversarial image generation. The task was to take any image and perform the required modifications to fool the Huggin Face access system and gain access. The system provides us with a
CSV file containing the embeddings for each employee. The system works by calculating the MSE(mean-squared error) of each row of the embeddings to the embeddings of the uploaded picture and if the error is less
than 1e-3, then it allows access. 

**Key concepts that I learned:**
1. ResNet-18 and Image Embeddings
ResNet-18 is a pretrained neural network that takes an image and outputs a 512-dimensional embedding (basically, a numerical representation of the image).
We remove the last fully connected layer so that the model only extracts features, without classifying the image.
These embeddings can be used for comparing images. If two images have similar embeddings, they are likely visually similar.

2. Mean Squared Error (MSE)
MSE measures how different two embeddings are.
A smaller MSE means the embeddings are more similar.
We modify the input image until its MSE with the target embedding is less than 1e-3.

3. Image Processing using PyTorch
We use PyTorch’s torchvision to resize, normalize, and convert images to tensors before passing them into ResNet.
The modified image is stored as a tensor and updated using gradient descent.

4. Gradient Descent for Image Modification
Instead of modifying embeddings, we directly modify the input image so that its embedding gets closer to the target.
The image tensor is treated as a variable, and gradients are computed using autograd.
This is the area where the key concept of backpropagation is used to efficiently calculate the derivatives.
The input image itself is treated as a variable and is updated using gradients.
The gradient tells us which direction to move the image pixels to make the embedding closer to the target.
We update the image step by step using Adam optimizer.
This optimizer helps us tweak the learning rate or the step size according to the loss function that is calculated.


**How the Code Works:**
Load the pre-trained ResNet-18 model (without the classification layer).

Read the target embeddings from a CSV file and store them in a dictionary.

Load an image, convert it into a tensor, and enable gradient updates.

Iteratively modify the image using gradient descent until its embedding is similar to the target embedding (MSE < 1e-3).

Save the modified image when the condition is met.



**The mathematics:**

![Gradient Descent](https://github.com/user-attachments/assets/3b1d8a2a-831f-48a2-a46a-f31015407ff0)
![Back](https://github.com/user-attachments/assets/817561dd-f98d-4fc3-bcea-d338c331a649)
