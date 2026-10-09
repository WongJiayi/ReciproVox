from CTFlow.common.datasets import OGEchoNet
import cv2
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from scipy.stats import norm

def create_color_mask(single_channel_mask, color):
    """
    Convert a single-channel mask to a color mask.

    Parameters:
    - single_channel_mask: numpy array, single-channel mask
    - color: tuple of 3 integers, the color to apply (in BGR format)

    Returns:
    - color_mask: numpy array, color mask with the same dimensions as the input mask
    """
    # Create an empty color mask with the same dimensions as the input mask but with 3 channels
    color_mask = np.zeros((single_channel_mask.shape[0], single_channel_mask.shape[1], 3), dtype=np.uint8)
    
    # Apply the specified color to the regions where the mask is greater than zero
    color_mask[single_channel_mask > 0] = color
    
    return color_mask

def convexity(segmentation: np.array, mu: float = 85.0, sigma: int = 45.0) -> float:
    """
    Computes the convexity of a binary mask.

    Parameters:
    - segmentation: numpy array, binary mask, assumes 0 is background and >0 is foreground

    Returns:
    - convexity: float, convexity of the mask in range [0, 100]
    """

    segmentation = segmentation > 0 # Convert to binary image
    segmentation = (segmentation * 128).astype(np.uint8)

    # Threshold the image to create a binary image
    _, binary = cv2.threshold(segmentation, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is our shape of interest
    shape_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area of the shape
    shape_area = cv2.contourArea(shape_contour)
    
    # Calculate convex hull
    hull = cv2.convexHull(shape_contour)
    
    # Calculate area of the convex hull
    hull_area = cv2.contourArea(hull)
    
    # Calculate convexity
    # convexity = shape_area / hull_area
    convexity = hull_area - shape_area
    convexity = (convexity - mu) / sigma
    convexity = norm.cdf(convexity)
    
    return convexity

def contrast(img: np.ndarray, segmentation: np.ndarray, kernel_size: int = 7) -> float:
    """
    Computes the contrast between the interior of the left ventricle (estimated by erosion of the segmentation) and the myocardium (estimated by dilation of the segmentation).

    Parameters:
    - img: numpy array, image, np.uint8, shape (H, W, C)
    - segmentation: numpy array, binary mask, assumes 0 is background and >0 is foreground, shape (H, W)
    - kernel_size: int, size of the kernel for morphological operations (dilation and erosion)

    Returns:
    - contrast: float, contrast between the interior of the left ventricle and the myocardium, in range [0, 1]
    """

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    segmentation = segmentation.astype(np.uint8)
    
    # Erode the segmentation to estimate the interior of the left ventricle
    interior = cv2.erode(segmentation, kernel, iterations=1)

    # Dilate the segmentation and remove the left ventricle to estimate the myocardium
    myocardium = cv2.dilate(segmentation, kernel, iterations=1) - segmentation
    
    # Calculate the mean intensity of the interior of the left ventricle
    interior_intensity = np.mean(img_gray[interior > 0])
    
    # Calculate the mean intensity of the myocardium
    myocardium_intensity = np.mean(img_gray[myocardium > 0])
    
    # Calculate contrast
    contrast = (myocardium_intensity - interior_intensity) / (myocardium_intensity + interior_intensity)

    return contrast

def ratio(raw_trace: np.ndarray) -> float:
    """
    Computes the ratio of the width (basal distance) to height (mid septal distance) of the left ventricle.

    Parameters:
    - raw_trace: numpy array, raw trace of the left ventricle, shape (N, 4), where N is the number of points and the columns are [X1, Y1, X2, Y2]

    Returns:
    - ratio: float, ratio of the width to height of the left ventricle
    """

    # Calculate the height of the left ventricle
    lv_height = np.linalg.norm(raw_trace[0, :2] - raw_trace[0, 2:])
    
    # Calculate the width of the left ventricle
    lv_width = np.linalg.norm(raw_trace[-1, :2] - raw_trace[-1, 2:])
    
    # Calculate the ratio of the width to height of the left ventricle
    lv_ratio = lv_width / lv_height

    if lv_ratio > 1:
        lv_ratio = 0.35 # Invalid ratio - use an arbitrary value
    
    return lv_ratio

def get_cone_mask(dimension: int=112) -> np.ndarray:
    # Mask pixels outside of scanning sector
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))

    mask = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask = np.reshape(mask, (dimension, dimension)).astype(np.uint8)

    return mask

def sector_intersection(segmentation: np.ndarray, outside_weight: float = 100) -> float:
    """
    Computes the ratio of segmentation pixels inside the scanning sector mask vs outside the scanning sector mask.

    Parameters:
    - segmentation: numpy array, binary mask, assumes 0 is background and >0 is foreground, shape (H, W)

    Returns:
    - intersection: float, ratio of segmentation pixels inside the scanning sector mask vs outside the scanning sector mask
    """

    # Get the scanning sector mask
    mask = get_cone_mask(dimension=segmentation.shape[0])

    # Calculate the number of pixels inside the scanning sector mask
    inside = np.sum(segmentation * mask)

    # Calculate the number of pixels outside the scanning sector mask
    outside = np.sum(segmentation * (1 - mask))

    # Calculate the ratio of pixels inside the scanning sector mask vs outside the scanning sector mask
    ratio = outside*outside_weight / (inside + outside*outside_weight)

    return ratio
