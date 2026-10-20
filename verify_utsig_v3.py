"""High-resolution monochrome scan engine calibrated on person-held-out UTSig.""" # Docstring = The official description stored for this module

import math
import statistics

import torch

from scripts.train_transfer_head import PairHead
from src.models.transfer_siamese import MobileNetSignatureEncoder
from verify import load_image


MANUAL_DIFFERENT_MAX = 0.37 # Manual comparison scores at or below this boundary represent no match
MANUAL_SAME_MIN = 0.76 # Manual comparison scores at or above this boundary represent a match
REGISTERED_DIFFERENT_MAX = 0.45 # Registered comparison has its own no match boundary because several references are combined
REGISTERED_SAME_MIN = 0.61 # Registered comparison needs this score to count one reference as match support

_encoder = None # Encoder = Converts a signature image into an embedding vector
_head = None # Head = Compares two embeddings and produces one raw similarity score


def get_models():
    global _encoder, _head # global allows this function to store the loaded models in the module variables

    if _encoder is None:
        _encoder = MobileNetSignatureEncoder(
            projection = False, # Keeps the original MobileNet embedding instead of adding a projection layer
            freeze_backbone = True # Keeps pretrained MobileNet weights unchanged
        ).cpu().eval() # Loads the encoder on CPU in evaluation mode

        _head = PairHead().cpu().eval() # PairHead turns two embeddings into one comparison score

        _head.load_state_dict(torch.load(
            "models/utsig_v3_binary_head.pth",
            map_location = "cpu" # Loads saved tensors directly into CPU memory
        )) # state_dict = Learned weights and buffers of the comparison head

    return _encoder, _head

def pair_probability(path_1, path_2):
    encoder, head = get_models() # Reuses the cached models instead of loading them for every comparison

    with torch.no_grad(): # Stops gradient tracking because verification does not update the models
        first = encoder.forward_once(load_image(path_1).cpu()) # first = Embedding vector of the first signature
        second = encoder.forward_once(load_image(path_2).cpu()) # forward_once = Runs one side of the Siamese encoder

        return torch.sigmoid(head(first, second)).item() # Sigmoid converts the raw head score into a probability between 0 and 1

def classify(probability, registered = False):
    low = REGISTERED_DIFFERENT_MAX if registered else MANUAL_DIFFERENT_MAX # low = Selected no-match boundary
    high = REGISTERED_SAME_MIN if registered else MANUAL_SAME_MIN # high = Selected match boundary

    if probability >= high:
        return "match"

    if probability <= low:
        return "no_match"

    return "uncertain"

def confidence(probability, registered = False):
    low = REGISTERED_DIFFERENT_MAX if registered else MANUAL_DIFFERENT_MAX
    high = REGISTERED_SAME_MIN if registered else MANUAL_SAME_MIN

    if probability >= high:
        return round(70 + min((probability - high) / (1 - high), 1) * 30, 1) # Maps the match area from 70% to 100%

    if probability <= low:
        return round(70 + min((low - probability) / max(low, 1e-6), 1) * 30, 1) # Maps the no-match area from 70% to 100% ------> 1e-6 prevents division by zero

    midpoint = (low + high) / 2 # Midpoint = Center of the uncertain probability range

    return round(50 + abs(probability - midpoint) / ((high - low) / 2) * 15, 1) # Uncertain confidence = 50% at the center and 65% near either boundary


def decide_registered(probabilities):
    values = sorted(float(value) for value in probabilities) # values = Reference probabilities converted to numbers and sorted from low to high
    median = statistics.median(values) # Median = Middle score ------> less affected by one unusually high or low reference

    required = max(1, math.ceil(len(values) * 0.60)) # required = At least 60% of references rounded upward and never less than one
    support = sum(value >= REGISTERED_SAME_MIN for value in values) # support = References with a score strong enough to vote for a match

    if support >= required and median >= REGISTERED_SAME_MIN:
        verdict = "match"
    elif median <= REGISTERED_DIFFERENT_MAX:
        verdict = "no_match"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict, # Final text decision
        "probability": median, # Stable group score used for the decision
        "support_count": support, # Number of registered signatures that support a match
        "required_support": required, # Minimum support needed for a match
    }
