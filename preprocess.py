import cv2 # It reads and processes the image
import numpy as np # A basic library for numerical operations and arrays

TARGET_SIZE = 224
INNER_SIZE = 190
PADDING_RATIO = 0.08
MIN_COMPONENT_AREA = 8 # Smaller than 8 px gets cleaned

LINE_LEN_RATIO = 0.45 # Is it long enough to be a border
BORDER_BAND = 0.12 # Border ratio at the image
BORDER_DEBRIS_RATIO = 0.08 # Is it small enough comparing to the signature, main element

def remove_border_lines(binary): # Removes the long horizontal and vertical frame lines at the edge of the image
    h, w = binary.shape # Gets the image size as height and width

    band_y = int(h * BORDER_BAND)
    band_x = int(w * BORDER_BAND)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, # Let it be rectangular in shape
        (max(int(w * LINE_LEN_RATIO), 1), 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, # Let it be rectangular in shape
        (1, max(int(h * LINE_LEN_RATIO), 1)),
    )

    horizontal_lines = cv2.morphologyEx(binary,cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    h_mask = np.zeros_like(binary) # Everything will be 0 so black at first

    h_mask[:band_y, :] = 255
    h_mask[h - band_y:, :] = 255


    v_mask = np.zeros_like(binary)

    v_mask[:, :band_x] = 255
    v_mask[:, w - band_x:] = 255


    to_remove = cv2.bitwise_or(
        cv2.bitwise_and(horizontal_lines, h_mask),
        cv2.bitwise_and(vertical_lines, v_mask)
    )

    return cv2.subtract(binary, to_remove)

def remove_border_debris(binary): # Small, non signature smudges on the edge are getting erased
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity = 8)

    # num_labels: Total number of components with including background
    # labels: A map showing which component each pixel belongs to
    # stats: The location, width, height and area of ​​each component
    # _: Center coordinates of the components, but _ is written here because it is not used

    if num_labels <= 1: # No white peace to delete could be found, just the background exits therefore exit
        return binary

    h, w = binary.shape

    largest_area = max( # It finds the largest area among the white components in the image since the max can be the signature
        stats[i, cv2.CC_STAT_AREA] # It gives how much of the i numbered element has white pixels: CC: connected components, STAT: Statistics, AREA is again area hand made by cv2

        for i in range(1, num_labels) # Starts from 1 since 0 is the black background and did not want to include that
    )

    band_y = int(h * BORDER_BAND)
    band_x = int(w * BORDER_BAND)

    cleaned = np.zeros_like(binary) # A new image is created that is all 0 so black so that the ones who are wanted to be kept will be written as white, 255

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT] # The coordinates of the left corner of the box
        y = stats[i, cv2.CC_STAT_TOP] # The coordinates of the top corner of the box

        cw = stats[i, cv2.CC_STAT_WIDTH] # Component width
        ch = stats[i, cv2.CC_STAT_HEIGHT] # Component height

        area = stats[i, cv2.CC_STAT_AREA] # The component's white pixel count -> cw x ch would not be able to give the real area, white area gives the real area of the signature

        in_band = (y + ch <= band_y) or (y >= h - band_y) or (x + cw <= band_x) or (x >= w - band_x) # It looks whether if the small piece is at the edge of the image or not

        if in_band and area < BORDER_DEBRIS_RATIO * largest_area:
            continue

        cleaned[labels == i] = 255 # Adds it to clean, gets protected

    return cleaned

def remove_small_components(binary): # Removes the small components without looking at the location
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity = 8)

    # num_labels: Total number of components with including background
    # labels: A map showing which component each pixel belongs to
    # stats: The location, width, height and area of ​​each component

    cleaned = np.zeros_like(binary)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 255

    return cleaned

def place_on_white_canvas(binary_signature, resize_threshold = 40, apply_close = True):
    # binary_signature: The cleaned up black and white signature image
    # resize_threshold = 40: Determines which of the gray pixels that appear after resizing will be considered the signature -> 0 ------------40-------------255
                        #     Background     Threshold    Signature

    # apply_close=True specifies that a morphological closing operation will be applied to fill very small gaps and connect broken lines in the signature it does not close large gaps it only attempts to repair small breaks of a few pixels in size

    h, w = binary_signature.shape

    scale = min(INNER_SIZE / w, INNER_SIZE / h) # INNER_SIZE = 190 if min was not used one side could exceed 190 now the signature is inside of 190 x 190 only one value of scale is chosen so that the signature keeps its ratio

    new_w = max(1, int(w * scale)) # max is used since it cannot be 0 px
    new_h = max(1, int(h * scale)) # Scale was found and finds the new values of width and height using it now

    resized = cv2.resize( # resized = The image that has been created
        binary_signature, # The image that will get resized
        (new_w, new_h), # target sizing
        interpolation = cv2.INTER_AREA # The methodology of resizing it calculates the values that the pixels will get after resizing
    ) # cv2.resize(src, dsize, interpolation = ...)

    _, resized = cv2.threshold( # It gets the grey colors back to black and white
        resized, # src
        resize_threshold, # 40
        255, # maxval, the ones that will be white are going to be 255 as value
        cv2.THRESH_BINARY # Binary thresholding process
    )

    if apply_close: # Nearby strokes in thick signatures may merge
        resized = cv2.morphologyEx(
            resized, # src
            cv2.MORPH_CLOSE, # op, dilation + erosion
            np.ones((2, 2), np.uint8) # kernel -> 2 x 2 sized every value of it is 1 and every value is an 8 bit unsigned integer from 0 to 255
        )

    canvas = np.ones((TARGET_SIZE, TARGET_SIZE), dtype = np.uint8) * 255 # 255 turns all the px to white canvas is the white baseline where the signature is going to take its place

    x_offset = (TARGET_SIZE - new_w) // 2 # The space it is going to take place in the x axis among the corner and the signature from both sides (target size - new size) gives the space and each corner is going to get the half of it
    y_offset = (TARGET_SIZE - new_h) // 2 # Same as the y axis // is used because a pixel position has to be an integer

    ink = resized > 0 # Is the value of the pixel higher than 0 if yes there is a signature now the matrix does not have values between 0 and 255 instead it has true or false

    selected_area = canvas[ # canvas is all white by 224 x 224
        y_offset:y_offset + new_h, # [Starting point, ending point]
        x_offset:x_offset + new_w
    ]

    selected_area[ink] = 0 # Turn the pixels that has the siganture to black

    return canvas

PREPROCESS_MODES = {
    "default": {
        "resize_threshold": 40,
        "apply_close": True # Small line breaks are attempted to be filled in
    },
    "sharp": {
        "resize_threshold": 150,
        "apply_close": True
    },
}

def preprocess_signature(input_path, output_path, mode = "default"):
    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError("Image could not be read")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0) # A single dark spot is softened because the surrounding area is white

    border_margin = int(
        min(gray.shape[0], gray.shape[1]) * 0.02
    )

    if border_margin > 0: # if the cleaning will be done in min 1 px so that if the margin is 0 prevents through whitening all
        gray[:, :border_margin] = 255
        gray[:, -border_margin:] = 255
        gray[:border_margin, :] = 255
        gray[-border_margin:, :] = 255

    _, binary = cv2.threshold(
        gray, # The image that will get processed
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary = remove_border_lines(binary)
    binary = remove_small_components(binary)
    binary = remove_border_debris(binary)

    points = cv2.findNonZero(binary)

    if points is None:
        blank = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
        cv2.imwrite(str(output_path), blank)
        return output_path

    x, y, w, h = cv2.boundingRect(points)

    padding = int(max(w, h) * PADDING_RATIO)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, binary.shape[1])
    y2 = min(y + h + padding, binary.shape[0])

    cropped_binary = binary[y1:y2, x1:x2]

    canvas_params = PREPROCESS_MODES.get(mode, PREPROCESS_MODES["default"])
    processed = place_on_white_canvas(cropped_binary, **canvas_params)

    cv2.imwrite(str(output_path), processed)

    return output_path

def preprocess_utsig_signature(input_path, output_path):
    image = cv2.imread(str(input_path))

    if image is None:
        raise ValueError("Image could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    binary = remove_small_components(binary)

    # UTSig zaten yakın ve düzenli olduğu için crop yapılmıyor.
    processed = place_on_white_canvas(binary)

    cv2.imwrite(str(output_path), processed)

    return output_path
