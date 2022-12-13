import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For static images:
IMAGE_FILES = ['images/train.jpg']
BG_COLOR = (0, 0, 0)  # black
MASK_COLOR = (255, 255, 255)  # white
with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=0) as selfie_segmentation:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        results = selfie_segmentation.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.4
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        output_image = np.where(condition, fg_image, bg_image)
        gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = image.copy()
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

        cv2.imshow('img', image_copy)
        cv2.waitKey(0)
        # cv2.imwrite('result/imgresult1.jpg', image_copy)
