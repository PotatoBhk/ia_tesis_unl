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
      rotated = False
      n = np.round(random(), 2)
      res = image
      if n >= 0.25 and n < 0.50:
        res = self.adjust_gamma(image, (random()*0.3))
      elif n < 0.25:
        res = self.rotate_image(image)
        rotated = True
      return (res, rotated)