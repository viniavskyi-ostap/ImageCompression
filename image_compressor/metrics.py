import numpy as np
from skimage.measure import compare_ssim


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
    WINDOW_SIZE = 7

    def is_similar(self, image, U, singular_values, V, rank):
        decompressed = (U[:, :rank] @ np.diag(singular_values[:rank]) @ V[:rank, :]).astype(np.float32)
        m, n = image.shape
        windows_number = 0
        simmilarity = 0
        for y in range(0, m - SSIMetric.WINDOW_SIZE, SSIMetric.WINDOW_SIZE):
            for x in range(0, n - SSIMetric.WINDOW_SIZE, SSIMetric.WINDOW_SIZE):
                windows_number+=1
                simmilarity += SSIMetric.ssim(image[y:y+SSIMetric.WINDOW_SIZE, x:x+SSIMetric.WINDOW_SIZE],
                                              decompressed[y:y+SSIMetric.WINDOW_SIZE, x:x+SSIMetric.WINDOW_SIZE])
        simmilarity /= windows_number
        print(rank, simmilarity)
        return simmilarity >= self._threshold

    # def is_similar(self, image, U, singular_values, V, rank):
    #     decompressed = (U[:, :rank] @ np.diag(singular_values[:rank]) @ V[:rank, :]).astype(np.float32)
    #     simmilarity = compare_ssim(image, decompressed, win_size=SSIMetric.WINDOW_SIZE)
    #     print(rank, simmilarity)
    #     return simmilarity > self._threshold

    @staticmethod
    def ssim(X, Y):
        mu1, mu2 = np.average(X), np.average(Y)
        luminance = (2 * mu1 * mu2 + SSIMetric.__c1) / (mu1 ** 2 + mu2 ** 2 + SSIMetric.__c1)
        std1, std2 = np.std(X), np.std(Y)
        contrast = (2 * std1 * std2 + SSIMetric.__c2) / (std1 ** 2 + std2 ** 2 + SSIMetric.__c2)
        covariance = np.matmul((X - mu1).flatten(), (Y - mu2).flatten()) / X.size
        structure = (covariance + SSIMetric.__c3) / (std1 * std2 + SSIMetric.__c3)
        return luminance * contrast * structure
