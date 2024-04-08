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

    # Estimate bounding box dimensions
    covariance_matrix = np.array([[M['mu20'], M['mu11']],
                                  [M['mu11'], M['mu02']]]) / M["m00"]
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    width = int(2 * np.sqrt(5.991 * eigenvalues[0]))  # 5.991 is the chi-square value for 95% confidence level
    height = int(2 * np.sqrt(5.991 * eigenvalues[1]))

    return cX, cY, width, height


def overlay(image, mask, color, alpha, resize=None):

    """
    Overlays a segmentation mask on an image.

    :param image: target RGB image
    :param mask: binary segmentation mask
    :param color: overlay color
    :param alpha: overlay color alpha value (opaqueness) in the range [0,1]
    :param resize: factor to resize the resulting image
    :return: RGB image with the segmentation mask overlay on the image
    """

    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose((1, 2, 0)), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def compute_roi(img_shape, centroid, bbox_shape, params, plot=True, target=None):

    """
    Computes and plots the bounding box corners for the regions of interest in the binary mask.

    :param img_shape: shape of the image
    :param centroid: tuple (cx, cy) of the centroid coordinates
    :param bbox_shape: tuple (width,height) of the body box dimensions
    :param params: Parameters() object containing the orientation and action box dimensions
    :param plot: whether to plot the boxes on top of the frame
    :param target: target RGB image to plot on
    :return: body, kick and punch bounding box corners
    """

    if plot:
        assert target is not None

    y_dim, x_dim = img_shape[0:2]
    width, height = bbox_shape
    cx, cy = centroid

    # Body box
    body_box_corner = (max(cx - width // 2, 0), max(cy - height // 2, 0))
    bbox_width = min(width, x_dim - 1 - body_box_corner[0])
    bbox_height = min(height, y_dim - 1 - body_box_corner[1])

    if plot:
        cv2.rectangle(target, pt1=body_box_corner[:2],
                      pt2=(body_box_corner[0] + bbox_width, body_box_corner[1] + bbox_height),
                      color=(0, 255, 0),
                      shift=0)

    act_box_width = int(params['act_box_width'] * x_dim)
    act_box_height = int(bbox_height // 3)
    spacing = int(params['spacing'] * x_dim)

    if params['orientation'] == "right":
        punch_box_corner = (min(cx + spacing, x_dim - 1), body_box_corner[1])
        kick_box_corner = (min(cx + spacing, x_dim - 1), body_box_corner[1] + 2 * act_box_height)
        act_box_width = min(act_box_width, x_dim - 1 - punch_box_corner[0])
    else:
        punch_box_corner = (max(cx - act_box_width - spacing, 0), body_box_corner[1])
        kick_box_corner = (max(cx - act_box_width - spacing, 0), body_box_corner[1] + 2 * act_box_height)
        act_box_width = min(act_box_width, punch_box_corner[0])

    body_box = (body_box_corner[0], body_box_corner[1],
                body_box_corner[0] + bbox_width, body_box_corner[1] + bbox_height)
    punch_box = (punch_box_corner[0], punch_box_corner[1],
                 punch_box_corner[0] + act_box_width, punch_box_corner[1] + act_box_height)
    kick_box = (kick_box_corner[0], kick_box_corner[1],
                kick_box_corner[0] + act_box_width, kick_box_corner[1] + act_box_height)

    # Adjust kick box based on player being bent down or not
    if plot:
        cv2.rectangle(target, pt1=punch_box[:2],
                      pt2=(punch_box[0] + act_box_width, punch_box[1] + act_box_height),
                      color=(0, 255, 0),
                      shift=0)

        cv2.rectangle(target, pt1=kick_box[:2],
                      pt2=(kick_box[0] + act_box_width, kick_box[1] + act_box_height),
                      color=(0, 255, 0),
                      shift=0)

    return body_box, punch_box, kick_box


def optical_flow_visualization(opt_flow):

    """
    Converts optical flow to RGB visualization image.

    :param opt_flow: input optical flow
    :return: RGB frame of optical flow visualization
    """

    hsv_visual = np.zeros((opt_flow.shape[0], opt_flow.shape[1], 3), dtype=np.uint8)
    hsv_visual[..., 1] = 255
    mag, ang = cv2.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])
    mag = np.where(mag > 2, mag, 0)
    hsv_visual[..., 0] = ang * 180 / np.pi / 2
    hsv_visual[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    visual = cv2.cvtColor(hsv_visual, cv2.COLOR_HSV2BGR)
    return visual
