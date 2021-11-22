from os import path
import cv2

class Utils:

    def path_exists(self, root):
        return path.exists(root)

    def join_path(self, root, file):
        return path.join(root,file)

    def preprocess_img(self, img, shape = (300,300)):
        frame_resized = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
        return cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 
                    (127.5, 127.5, 127.5), False)