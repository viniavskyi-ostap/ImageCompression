import numpy as np
from image_compressor.svd import svd
from image_compressor.compressed import Compressed
import time


class SVDCompressor:
    def __init__(self, metric):
        self.__metric = metric

    def compress(self, image):
        image = image.astype(np.float32)
        m, n = image.shape
        transposed = False
        if n > m:
            image = image.T
            transposed = True
        U, S, V = svd(image)
        if n > m:
            U, V = V.T, U.T

        singular_values = np.diag(S)
        indexes = np.argsort(singular_values)[::-1]
        U = U[:, indexes]
        V = V[indexes, :]
        singular_values = singular_values[indexes]

        if transposed:
            image = image.T
        rank = 1
        while not self.__metric.is_similar(image, U, singular_values, V, rank):
            rank += 1

        return Compressed(U[:, :rank], singular_values[:rank], V[:rank, :])

