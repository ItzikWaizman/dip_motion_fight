import cv2


def get_centroid(frame):
    M = cv2.moments(frame)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = frame.shape[1] // 2
        cY = frame.shape[0] // 2
    return cX, cY
