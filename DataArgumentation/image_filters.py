import numpy as np
import cv2

def change_brightness(img: np.ndarray, beta: float) -> np.ndarray:
    """
    Adjusts the contrast and brightness of an image.
    
    Args:
        img (np.ndarray): Input image
        beta (float): Brightness adjustment value (>0: increase brightness, <0: decrease brightness)
    
    Returns:
        np.ndarray: Image with adjusted contrast and brightness
    """
    filtered_image = cv2.convertScaleAbs(img, alpha=1, beta=beta)
    return filtered_image

def change_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """
    Adjusts the contrast of an image.
    
    Args:
        img (np.ndarray): Input image
        alpha (float): Contrast adjustment value (>1: increase contrast, <1: decrease contrast)
    
    Returns:
        np.ndarray: Image with adjusted contrast
    """
    filtered_image = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    return filtered_image

def change_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Adjusts the gamma of an image.
    
    Args:
        img (np.ndarray): Input image
        gamma (float): Gamma correction value (>1: darker, <1: brighter)
    
    Returns:
        np.ndarray: Image with adjusted gamma
    """
    filtered_image = np.array(255 * (img / 255)**(1 / gamma), dtype=np.uint8)
    return filtered_image

def change_saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Adjusts the saturation of an image.
    
    Args:
        img (np.ndarray): Input image
        scale (float): Saturation scale factor (>1: increase saturation, <1: decrease saturation)
    
    Returns:
        np.ndarray: Image with adjusted saturation
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 1] = hsv_img[:, :, 1] * scale
    filtered_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return filtered_image

def change_hue(img: np.ndarray, shift: float) -> np.ndarray:
    """
    Shifts the hue of an image.
    
    Args:
        img (np.ndarray): Input image
        shift (float): Hue shift value in degrees (0-180)
    
    Returns:
        np.ndarray: Image with shifted hue
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] + shift) % 180
    filtered_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return filtered_image

def make_blur(img: np.ndarray, sigma: float, kernel_shape: tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Applies Gaussian blur to an image.
    
    Args:
        img (np.ndarray): Input image
        sigma (float): Standard deviation for Gaussian kernel
        kernel_shape (tuple[int, int], optional): Size of the kernel. Defaults to (5, 5)
    
    Returns:
        np.ndarray: Blurred image
    """
    filtered_image = cv2.GaussianBlur(img, kernel_shape, sigma)
    return filtered_image

def make_sharpen(img: np.ndarray, repeat: int) -> np.ndarray:
    """
    Sharpens an image using a kernel filter.
    
    Args:
        img (np.ndarray): Input image
        repeat (int): Number of times to apply the sharpening filter
    
    Returns:
        np.ndarray: Sharpened image
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filtered_image = img
    for _ in range(repeat):
        filtered_image = cv2.filter2D(filtered_image, -1, kernel)
    return filtered_image