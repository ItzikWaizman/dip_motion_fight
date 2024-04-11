import queue
import cv2
import numpy as np
from threading import Thread
import utils
import time


class FrameReader:

    """
    Responsibilities: Read frames from the camera and perform initial processing - mainly player segmentation.
    The processed frames will be stored in a queue which will serve the consumer for movement analysis.
    """

    def __init__(self, params):
        self.optflow_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(0)
        self.prev_frame = None
        self.fps = 0
        self.display = params['display_preprocess']
        self.params = params

    def start_capture(self):
        th = Thread(target=self.read_frames, daemon=True)
        th.start()
        return th

    # noinspection PyTypeChecker
    def read_frames(self):
        i = 0
        ret, self.prev_frame = self.capture.read()
        if self.params['resize']:
            self.prev_frame = cv2.resize(self.prev_frame, (0, 0),
                                         fx=self.params['resize_factor'], fy=self.params['resize_factor'])
        start_time = time.time()
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            if self.params['resize']:
                frame = cv2.resize(frame, (0, 0),
                                   fx=self.params['resize_factor'], fy=self.params['resize_factor'])

            # Throw out identical frames
            if np.max(np.abs(frame - self.prev_frame)) == 0:
                continue

            # Compute difference frame
            diff = np.zeros_like(frame)
            cv2.absdiff(frame, self.prev_frame, diff)
            diff = np.mean(diff, axis=2)
            diff_mask = cv2.threshold(diff, self.params['mag_thresh'], 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            diff_mask = cv2.medianBlur(diff_mask, 3)
            diff_edges = cv2.Canny(diff_mask, threshold1=50, threshold2=150, apertureSize=3)
            frame_copy, circles = self.detect_circles(np.copy(frame), diff_edges)

            # Compute optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            opt_flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, flow=None, pyr_scale=0.3, levels=3,
                                                    winsize=15, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)

            self.prev_frame = frame

            # Equeue optical flow
            if self.optflow_buffer.full():
                self.optflow_buffer.get()
            self.optflow_buffer.put(opt_flow)

            # Display segmentation and background
            if self.display:
                cv2.imshow("Edges of difference frame", diff_edges)
                cv2.imshow("Circles Found", frame_copy)

                # Visualize optical flow
                hsv_visual = np.zeros((opt_flow.shape[0], opt_flow.shape[1], 3), dtype=np.uint8)
                hsv_visual[..., 1] = 255
                mag, ang = cv2.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])
                mag = np.where(mag > 2, mag, 0)
                hsv_visual[..., 0] = ang * 180 / np.pi / 2
                hsv_visual[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                visual_optflow = cv2.cvtColor(hsv_visual, cv2.COLOR_HSV2BGR)
                cv2.imshow("Optical Flow", visual_optflow)

                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Update fps
            if i % 10 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                self.fps = int(10 / elapsed_time)
            i = i + 1

        self.capture.release()
        print("Frame Reader Terminated...")

    @staticmethod
    def detect_circles(frame, mask):

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=2, minDist=100, param1=50, param2=50,
                                   minRadius=20, maxRadius=60)

        detected_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            highest_circle = min(circles, key=lambda circle: circle[1])

            (x, y, r) = highest_circle

            # Draw the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            # Draw the center of the circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        else:
            highest_circle = None

        return frame, highest_circle

    def get_flow(self):
        if not self.optflow_buffer.empty():
            return self.optflow_buffer.get()
        return None

    def get_mask(self):
        return None
