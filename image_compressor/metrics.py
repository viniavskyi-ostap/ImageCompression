import numpy as np


class Metric:
    def __init__(self, threshold):
        self._threshold = threshold


class EnergyRatioMetric(Metric):
    def is_similar(self, image, U, singular_values, V, rank):
        original_energy = np.sum(singular_values ** 2)
        current_energy = np.sum(singular_values[:rank] ** 2)
        return current_energy / original_energy >= self._threshold


class SSIMetric(Metric):
    __L = 255
    __k1 = 0.01
    __k2 = 0.03
    __c1 = (__k1 * __L) ** 2
    __c2 = (__k2 * __L) ** 2
    __c3 = __c2 / 2

    def is_similar(self, image, U, singular_values, V, rank):
        edited = (U[:, :rank] @ np.diag(singular_values[:rank]) @ V[:rank, :]).astype(np.float32)
        simmilarity = self.ssim(image, edited)
        return simmilarity >= self._threshold

    def ssim(self, image, edited):
        mu1, mu2 = np.average(image), np.average(edited)
        luminance = (2 * mu1 * mu2 + self.__c1) / (mu1 ** 2 + mu2 ** 2 + self.__c1)
        std1, std2 = np.std(image), np.std(edited)
        contrast = (2 * std1 * std2 + self.__c2) / (std1 ** 2 + std2 ** 2 + self.__c2)
        covariance = np.matmul((image - mu1).flatten(), (edited - mu2).flatten()) / image.size
        structure = (covariance + self.__c3) / (std1 * std2 + self.__c3)
        return luminance * contrast * structure
