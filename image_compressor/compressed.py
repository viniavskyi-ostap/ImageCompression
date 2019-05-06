import numpy as np


class Compressed:
    def __init__(self, U, S, V):
        self.__U = U
        self.__S = S
        self.__V = V

    def to_image(self):
        S = np.diag(self.__S)
        return np.rint(self.__U @ S @ self.__V).astype(np.uint8)

    def get_effective_rank(self):
        return self.__S.size
