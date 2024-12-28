import numpy as np
import cv2
import matplotlib.pyplot as plt

import image_filters

if __name__ == "__main__":
    img = cv2.imread("data/images/kcity_1_frame0.png", cv2.IMREAD_COLOR)

    # Get base filename from input path
    base_filename = "kcity_1_frame0"

    # Brightness variations
    plt.figure(figsize=(25, 5))
    brightness_values = [-100, -50, 0, 50, 100]
    for i, brightness in enumerate(brightness_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.change_brightness(img, brightness)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Brightness: {brightness}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_brightness_comparison.png')
    plt.close()

    # Contrast variations  
    plt.figure(figsize=(25, 5))
    contrast_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    for i, contrast in enumerate(contrast_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.change_contrast(img, contrast)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Contrast: {contrast}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_contrast_comparison.png')
    plt.close()

    # Gamma variations
    plt.figure(figsize=(25, 5))
    gamma_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    for i, gamma in enumerate(gamma_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.change_gamma(img, gamma)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Gamma: {gamma}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_gamma_comparison.png')
    plt.close()

    # Saturation variations
    plt.figure(figsize=(25, 5))
    saturation_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for i, saturation in enumerate(saturation_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.change_saturation(img, saturation)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Saturation: {saturation}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_saturation_comparison.png')
    plt.close()

    # Hue variations
    plt.figure(figsize=(25, 5))
    hue_values = [0, 45, 90, 135, 180]
    for i, hue in enumerate(hue_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.change_hue(img, hue)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Hue: {hue}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_hue_comparison.png')
    plt.close()

    # Gaussian blur variations
    plt.figure(figsize=(25, 5))
    sigma_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    for i, sigma in enumerate(sigma_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.make_blur(img, sigma)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Blur σ: {sigma}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_blur_comparison.png')
    plt.close()

    # Sharpness variations
    plt.figure(figsize=(25, 5))
    sharpen_values = [0, 1, 2, 3, 4]
    for i, repeat in enumerate(sharpen_values, 1):
        plt.subplot(1, 5, i)
        filtered = image_filters.make_sharpen(img, repeat)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title(f'Sharpen x{repeat}', y=-0.2)
        plt.axis('off')
    plt.tight_layout(h_pad=3.0, w_pad=0.5)
    plt.savefig(f'{base_filename}_sharpen_comparison.png')
    plt.close()
