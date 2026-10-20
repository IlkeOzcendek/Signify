from pathlib import Path
from PIL import Image # Python Imaging Library; open, create, save or organise a photo

# import numpy as np # np is an allias, just to make shorter. You can use whatever you'd like actually however in general np is getting used
# import torch # All of torch is gotten

from torchvision import transforms

transform = transforms.Compose([ # Instead of running all above, compose does all everytime we need these
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_path = Path("dataset/cedar/signatures/full_org/original_1_1.png")

image = Image.open(image_path).convert("L") # Create an object of image and then make it all grey (L) toned (0 -> black, 255 -> white, some value between)

image_tensor = transform(image)

print("Tensor shape is ", image_tensor.shape)
print("Tensor min: ", image_tensor.min())
print("Tensor max: ", image_tensor.max())

# print("Original image type is ", type(image)) # It became pillow object since up line
# print("Original image mode is  ", image.mode) # Color format: L
# print("Original image size is ", image.size) # height and width of the image

# # image = image.resize((224, 224))

# print("The resized image size is ", image.size)

# image_array = np.array(image) # Convert the image to a NumPy array for image processing and machine learning, ML needs pixels to be able to work on the images

# print("Numpy array type is ", type(image_array)) # The data type, it should be numpy.ndarray
# print("Numpy array shape is ", image_array.shape) # Gives the size of the array (row x column)
# print("Min pixel value is ", image_array.min()) # Finds the smallest pixel value in the picture, for ex 0 = fully black
# print("Max pixel value is ", image_array.max()) # For ex 255, fully white

# # image_tensor = torch.tensor(image_array, dtype = torch.float32) # Numpy array -> PyTorch tensor; data type float and save it with 32 bits (64 would be too slow)

# print("Tensor shape before channel is ", image_tensor.shape)
# # image_tensor = image_tensor.unsqueeze(0) # Adds one channel to the very beginning as "1" since it was 2 dimentional and now we need 3 in CNNs for RGB so 1 channe + height + width = 3 dimentional
# print("The shape afterthe channel is ", image_tensor.shape)

# # image_tensor = image_tensor / 255.0 # All pixel values are between 0 and 1 since neural networks learns better using values that are between 0 and 1

# print("Tensor min: ", image_tensor.min()) # min can be 0.0
# print("Tensor max: ", image_tensor.max()) # max can be 1.0

# # Min - max normalisation done, the data is more appropriately scaled
