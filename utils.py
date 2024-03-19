import cv2
import numpy as np


def get_centroid(frame):
    """
    Finds the centroid of a binary frame.

    :param frame: target frame
    :return: Centroid of a binary frame
    """
    M = cv2.moments(frame)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = frame.shape[1] // 2
        cY = frame.shape[0] // 2
    return cX, cY


def overlay(image, mask, color, alpha, resize=None):
    """
    Overlays a segmentation mask on an image.

    :param image: target RGB image
    :param mask: binary segmentation mask
    :param color: overlay color
    :param alpha: overlay color alpha value (opaqueness) in the range [0,1]
    :return: RGB image with the segmentation mask overlay on the image.
    :param resize: factor to resize the resulting image
    """

    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
