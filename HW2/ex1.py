import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    # Need to implement here
    return np.pad(img, pad_width=filter_size//2, mode='edge')

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    result = np.zeros(img.shape)
    padded_img = padding_img(img, filter_size)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.mean(region)
    
    return result.astype(img.dtype)

def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    # Need to implement here
    result = np.zeros(img.shape)
    padded_img = padding_img(img, filter_size)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+filter_size, j:j+filter_size]
            result[i, j] = np.median(region)

    return result.astype(img.dtype)


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    # Need to implement here
    gt_img = gt_img.astype(np.float32)
    smooth_img = smooth_img.astype(np.float32)

    mse = np.mean((gt_img - smooth_img) ** 2)
    return 10 * np.log10(255**2/mse)

def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "./ex1_images/noise.png" # <- need to specify the path to the noise image
    img_gt = "./ex1_images/ori_img.png" # <- need to specify the path to the gt image
    img = read_img(img_noise)
    img_gt = read_img(img_gt)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img_gt, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img_gt, median_smoothed_img))

