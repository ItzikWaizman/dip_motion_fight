import cv2
import numpy as np


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
    body_box_corner = (max(cx - width // 2, 0), cy)
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

    punch_box_corner = (max(cx - act_box_width - spacing, 0), body_box_corner[1])
    kick_box_corner = (max(cx - act_box_width - spacing, 0), body_box_corner[1] + 2 * act_box_height)
    act_box_width = min(act_box_width, cx - spacing)

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


def dist(x1, y1, x2, y2):

    """
    Computes the distance between two points on a plane.

    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point

    :return: Euclidean distance between (x1, y1) and (x2, y2)
    """
    return np.linalg.norm([x2 - x1, y2 - y1])
