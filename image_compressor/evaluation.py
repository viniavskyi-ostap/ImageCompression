from math import log10, sqrt
import numpy as np


def mse(image, compressed):
    compressed_image = compressed.to_image()
    difference = abs(image - compressed_image)
    return sum(sum(difference)) / image.size


def psnr(image, compressed):
    max_i = 255
    error = mse(image, compressed)
    return 20 * log10(max_i / sqrt(error))


def pearson_correlation(image, compressed):
    compressed_image = compressed.to_image()
    mu1, mu2 = np.average(image), np.average(compressed_image)
    std1, std2 = np.std(image), np.std(compressed_image)
    covariance = np.matmul((image - mu1).flatten(), (compressed_image - mu2).flatten()) / image.size
    return covariance / (std1 * std2)


def cr(image, compressed):
    rank = compressed.get_effective_rank()
    return image.size / (rank * sum(image.shape))
