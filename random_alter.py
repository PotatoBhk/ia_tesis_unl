import cv2
import numpy as np

# generate random floating point values
from random import seed
from random import random

class ImageAlt():
    def __init__(self):        
        seed(1)

    def adjust_gamma(self, image, gamma=1.0):
      invGamma = 1.0 / gamma
      table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")

      return cv2.LUT(image, table)

    def rotate_image(self, image):
      image_rot = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      return image_rot

    def random_alter(self, image):
      modified = 0
      n = np.round(random(), 2)
      res = image
      if n >= 0.25 and n < 0.50:
        res = self.adjust_gamma(image, ((random() + 0.5) * 2))
        modified = 1
      elif n < 0.1:
        res = self.rotate_image(image)
        modified = 2
      return (res, modified)