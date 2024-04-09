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
        self.mask_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.optflow_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(0)  # Assuming default camera
        self.prev_gray = None
        self.fps = 0
        self.display = params['display_preprocess']
        self.params = params

    def start_capture(self):
        th = Thread(target=self.read_frames, daemon=True)
        th.start()
        return th

    def read_frames(self):
        i = 0
        ret, self.prev_gray = self.capture.read()
        self.prev_gray = cv2.cvtColor(self.prev_gray, cv2.COLOR_BGR2GRAY)
        if self.params['resize']:
            self.prev_gray = cv2.resize(self.prev_gray, (0, 0),
                                        fx=self.params['resize_factor'], fy=self.params['resize_factor'])
        start_time = time.time()
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            if self.params['resize']:
                frame = cv2.resize(frame, (0, 0),
                                   fx=self.params['resize_factor'], fy=self.params['resize_factor'])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.abs(np.max(gray - self.prev_gray)) == 0:
                continue

            # Compute optical flow
            mag = np.zeros_like(gray)
            cv2.absdiff(gray, self.prev_gray, mag)
            _, visual = cv2.threshold(mag, self.params['mag_thresh'], 255, cv2.THRESH_BINARY)
            visual = visual.astype(np.uint8)
            self.prev_gray = gray

            # compute fgMask
            fgMask = self.segment_player(mag)

            # Equeue foreground mask
            if self.mask_buffer.full():
                self.mask_buffer.get()
            self.mask_buffer.put(fgMask)

            # Display segmentation and background
            if self.display:

                segmented_frame = utils.overlay(image=frame, mask=fgMask, color=(255, 0, 0), alpha=0.7)
                segmented_frame = cv2.putText(img=segmented_frame, text=f"fps: {self.fps}", org=(15, 30),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                              color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow("Player Segmentation", segmented_frame)
                cv2.imshow("Optical Flow", visual)

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

    def segment_player(self, frame):

        _, fgMask = cv2.threshold(frame, self.params['mag_thresh'], 255, cv2.THRESH_BINARY)
        fgMask = fgMask.astype(np.uint8)

        # Erosion and dilation to remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.dilate(fgMask, kernel, iterations=2)

        for _ in range(3):
            # Filter out noise and smaller components, keeping the largest component
            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                fgMask = np.zeros_like(fgMask)

                cv2.drawContours(fgMask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            fgMask = cv2.dilate(fgMask, kernel, iterations=2)

        return fgMask

    def get_mask(self):
        if not self.mask_buffer.empty():
            return self.mask_buffer.get()
        return None
