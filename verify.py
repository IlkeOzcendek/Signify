import torch
import tempfile

from PIL import Image
from torchvision import transforms

from preprocess import preprocess_signature
from src.models.siamese_network import SiameseNetwork
from src.models.siamese_network_legacy import SiameseNetworkLegacy
from src.threshold import load_threshold


# MODEL SELECTION ------> Change both values and run evaluate.py to recalculate the threshold for another model
# Current model option ------> MODEL_PATH = "best_siamese_model.pth" | MODEL_CLASS = SiameseNetwork
MODEL_PATH = "models/last_model_cedar_selected_utsig.pth" # Path = Location of the saved model weights
MODEL_CLASS = SiameseNetworkLegacy # Class = Architecture that must match the saved weights ------> currently uses the legacy model


if torch.backends.mps.is_available():
    device = torch.device("mps") # Uses the Apple Silicon GPU when it is available
elif torch.cuda.is_available():
    device = torch.device("cuda") # Uses the NVIDIA GPU when it is available
else:
    device = torch.device("cpu") # Uses the processor when there is no supported GPU


transform = transforms.Compose([
    transforms.ToTensor() # Converts the PIL image into a tensor and scales pixels from 0-255 to 0-1
])


_model = None # Model cache = Keeps one loaded model in memory ------> several references do not reload it from disk


def get_model():
    global _model # global allows this function to store the loaded model in the module variable

    if _model is None:
        _model = MODEL_CLASS().to(device) # Creates the selected architecture and moves it to the selected device

        with torch.no_grad(): # Stops gradient tracking because this pass does not train the model
            dummy = torch.zeros(1, 1, 224, 224, device = device) # Dummy = Empty sample with shape [batch channel height width]
            _model(dummy, dummy) # Initializes the legacy LazyLinear input size before loading its saved weights

        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location = device) # map_location loads saved tensors directly onto the selected device
        ) # state_dict = Every learned weight and buffer in the model

        _model.eval() # Switches layers such as dropout and batch normalization to evaluation mode

    return _model


def load_image(image_path, preprocess_mode = "default"):
    with tempfile.NamedTemporaryFile(suffix = ".png") as temp: # Temporary file stores the processed image and is deleted automatically
        preprocess_signature(image_path, temp.name, mode = preprocess_mode) # Preprocessing cleans and standardizes the signature before inference

        image = Image.open(temp.name).convert("L") # L = One-channel grayscale image
        image = transform(image) # Shape changes from [height width] to [channel height width]
        image = image.unsqueeze(0) # Adds the batch dimension ------> [1 channel height width]
        image = image.to(device) # Image and model must be on the same device

    return image


def verify_signature(image_path_1, image_path_2, threshold = None,
                     preprocess_mode = "default", preprocess_mode_2 = None):
    if threshold is None:
        threshold = load_threshold() # Threshold = Saved decision boundary found by evaluate.py

    if preprocess_mode_2 is None:
        preprocess_mode_2 = preprocess_mode # Uses the same processing for both images unless a second mode is given
    # Different modes process the reference and questioned image asymmetrically ------> this can make a forgery distance artificially smaller

    model = get_model() # Reuses the model already stored in memory

    img1 = load_image(image_path_1, preprocess_mode = preprocess_mode) # img1 = Processed reference signature tensor
    img2 = load_image(image_path_2, preprocess_mode = preprocess_mode_2) # img2 = Processed questioned signature tensor

    with torch.no_grad(): # Verification only measures the model and never updates its weights
        output1, output2 = model(img1, img2) # Outputs = Embedding vectors that represent both signatures with numbers

        distance = torch.nn.functional.pairwise_distance(output1, output2) # Euclidean distance ------> smaller means the signatures are more similar

    distance_value = distance.item() # item turns the one-value distance tensor into a Python number

    if distance_value < threshold:
        result = "Genuine" # Distance below the boundary represents a matching signature
    else:
        result = "Forgery" # Distance at or above the boundary represents a different signature

    return result, distance_value
