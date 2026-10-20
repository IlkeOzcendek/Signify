"""Archive and low-resolution scan engine. Production v1 stays independent.""" # Docstring = The official description stored for this module

import math
import statistics

import torch

from src.models.archive_pair_model import ArchivePairHead
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.archive_grayscale_v2 import load_image as load_gray_image
from verify import load_image as load_binary_image

LOW_PROBABILITY = 0.62 # Scores at or below this boundary represent no match
HIGH_PROBABILITY = 0.71 # Scores at or above this boundary represent a match
GRAY_WEIGHT = 0.75 # Gives 75% of the final score to the grayscale model and 25% to the binary model

_encoder = None # Encoder = Converts a signature image into an embedding vector
_binary_head = None # Binary head = Compares signatures processed as black and white images
_gray_head = None # Gray head = Compares signatures while keeping grayscale information

def get_archive_models():
    global _encoder, _binary_head, _gray_head # global allows this function to store the loaded models in the module variables

    if _encoder is None:
        _encoder = MobileNetSignatureEncoder(
            projection = False, # Keeps the original MobileNet embedding instead of adding a projection layer
            freeze_backbone = True # Keeps pretrained MobileNet weights unchanged
        ).cpu().eval() # Loads the encoder on CPU in evaluation mode

        _binary_head = ArchivePairHead().cpu().eval()
        _gray_head = ArchivePairHead().cpu().eval()

        _binary_head.load_state_dict(torch.load(
            "models/archive_v2_binary_head.pth",
            map_location = "cpu" # Loads saved tensors directly into CPU memory
        )) # state_dict = Learned weights and buffers of the binary comparison head

        _gray_head.load_state_dict(torch.load(
            "models/archive_v2_gray_head.pth",
            map_location = "cpu"
        )) # Loads the learned grayscale comparison weights

    return _encoder, _binary_head, _gray_head

def archive_pair_probability(path_1, path_2):
    encoder, binary_head, gray_head = get_archive_models() # Heads share one encoder but compare two different image versions

    with torch.no_grad(): # Stops gradient tracking because verification does not update the models
        binary_1 = encoder.forward_once(load_binary_image(path_1).cpu()) # binary_1 = Embedding of the first black and white signature
        binary_2 = encoder.forward_once(load_binary_image(path_2).cpu()) # forward_once = Runs one side of the Siamese encoder

        gray_1 = encoder.forward_once(load_gray_image(path_1).unsqueeze(0)) # unsqueeze(0) adds the batch dimension expected by the encoder
        gray_2 = encoder.forward_once(load_gray_image(path_2).unsqueeze(0))

        binary_probability = torch.sigmoid(binary_head(binary_1, binary_2)).item() # Sigmoid converts the raw head score into a value between 0 and 1
        gray_probability = torch.sigmoid(gray_head(gray_1, gray_2)).item()

    return GRAY_WEIGHT * gray_probability + (1 - GRAY_WEIGHT) * binary_probability # Weighted score = 0.75 * gray + 0.25 * binary

def archive_verdict(probability):
    if probability >= HIGH_PROBABILITY:
        return "match"

    if probability <= LOW_PROBABILITY:
        return "no_match"

    return "uncertain"

def archive_confidence(probability):
    if probability >= HIGH_PROBABILITY:
        return round(70 + min((probability - HIGH_PROBABILITY) / (1 - HIGH_PROBABILITY), 1) * 30, 1) # Maps the match area from 70% to 100%

    if probability <= LOW_PROBABILITY:
        return round(70 + min((LOW_PROBABILITY - probability) / LOW_PROBABILITY, 1) * 30, 1) # Maps the no-match area from 70% to 100%

    midpoint = (LOW_PROBABILITY + HIGH_PROBABILITY) / 2 # Midpoint = Center of the uncertain probability range
    half = (HIGH_PROBABILITY - LOW_PROBABILITY) / 2 # Half = Distance from the midpoint to either boundary

    return round(50 + abs(probability - midpoint) / half * 15, 1) # Uncertain confidence = 50% at the center and 65% near either boundary

def decide_archive_registered(probabilities):
    ordered = sorted(float(value) for value in probabilities) # ordered = Pair probabilities converted to numbers and sorted from low to high
    count = len(ordered) # count = Number of registered signatures used in the decision

    median_probability = statistics.median(ordered) # Median = Middle score ------> less affected by one unusually high or low result
    required_support = max(1, math.ceil(count * 0.60)) # Requires at least 60% support and rounds the required count upward
    support_count = sum(value >= HIGH_PROBABILITY for value in ordered) # support_count = Scores strong enough to vote for a match

    if support_count >= required_support and median_probability >= HIGH_PROBABILITY:
        verdict = "match"
    elif median_probability <= LOW_PROBABILITY:
        verdict = "no_match"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict, # Final text decision
        "probability": median_probability, # Stable group score used for the decision
        "support_count": support_count, # Number of signatures that support a match
        "required_support": required_support, # Minimum support needed for a match
    }
