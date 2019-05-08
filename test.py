from image_compressor.svd_compressor import SVDCompressor
from image_compressor.metrics import EnergyRatioMetric, SSIMetric
import numpy as np
import cv2

image = cv2.imread("Lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

compressor = SVDCompressor(SSIMetric(0.93))
compressed = compressor.compress(image)
new_image = compressed.to_image()
print(compressed.get_effective_rank())
cv2.imshow("", np.hstack([image, new_image]))
cv2.waitKey(0)
