from image_compressor.svd_compressor import SVDCompressor
from image_compressor.metrics import EnergyRatioMetric, SSIMetric
from image_compressor.evaluation import mse, psnr, pearson_correlation, cr
import numpy as np
import cv2
import time

image = cv2.imread("image/download.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

compressor = SVDCompressor(SSIMetric(0.995))
compressed = compressor.compress(image)
new_image = compressed.to_image()
# print(compressed.get_effective_rank())
print(mse(image, compressed))
print(psnr(image, compressed))
print(pearson_correlation(image, compressed))
print(cr(image, compressed))


cv2.imshow("", np.hstack([image, new_image]))
cv2.waitKey(0)
