import numpy as np


def mse(image, compressed):
    image = image.astype(np.float)
    compressed_image = compressed.to_image().astype(np.float)
    difference = (image - compressed_image) ** 2
    return np.sum(difference) / image.size


def psnr(image, compressed):
    max_i = 255
    error = mse(image, compressed)
    return 20 * np.log10(max_i / np.sqrt(error)) if error > 0 else float("inf")


def pearson_correlation(image, compressed):
    compressed_image = compressed.to_image()
    mu1, mu2 = np.average(image), np.average(compressed_image)
    std1, std2 = np.std(image), np.std(compressed_image)
    covariance = np.matmul((image - mu1).flatten(), (compressed_image - mu2).flatten()) / image.size
    return covariance / (std1 * std2)


def cr(image, compressed):
    rank = compressed.get_effective_rank()
    return image.size / (2 * rank * np.sum(image.shape))
