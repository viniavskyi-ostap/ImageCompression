import numpy as np


class Compressed:
    MAX_PIXEL_INTENSITY = 255
    MIN_PIXEL_INTENSITY = 0

    def __init__(self, U, S, V):
        self.__U = U.astype(np.float16)
        self.__S = S.astype(np.float16)
        self.__V = V.astype(np.float16)

    def to_image(self):
        S = np.diag(self.__S)
        image = np.rint(self.__U @ S @ self.__V)
        image[image < Compressed.MIN_PIXEL_INTENSITY] = Compressed.MIN_PIXEL_INTENSITY
        image[image > Compressed.MAX_PIXEL_INTENSITY] = Compressed.MAX_PIXEL_INTENSITY
        return image.astype(np.uint8)

    def get_effective_rank(self):
        return self.__S.size
