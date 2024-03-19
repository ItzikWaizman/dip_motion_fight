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
        self.frame_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(0)  # Assuming default camera
        self.background = None
        self.fps = 0
        self.backSub = cv2.createBackgroundSubtractorKNN(history=params['history'],
                                                         dist2Threshold=params['dist2thresh'],
                                                         detectShadows=False)
        self.backSub.setkNNSamples(4)
        '''self.backSub = cv2.createBackgroundSubtractorMOG2(history=params['history'],
                                                          varThreshold=params['max_var'],
                                                          detectShadows=False)'''
        self.display = params['display_preprocess']
        self.params = params

    def start_capture(self):
        th = Thread(target=self.read_frames, daemon=True)
        th.start()
        return th

    def read_frames(self):
        i = 0
        start_time = time.time()
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            # Compute fgMask
            fgMask = self.segment_player(frame)

            # Equeue foreground mask
            if self.frame_buffer.full():
                self.frame_buffer.get()
            self.frame_buffer.put(fgMask)

            # Display segmentation and background
            if self.display:
                self.background = self.backSub.getBackgroundImage()
                if self.params['colorspace'] == "HSV":
                    self.background = cv2.cvtColor(self.background, cv2.COLOR_HSV2BGR)

                segmented_frame = utils.overlay(image=frame, mask=fgMask, color=(255, 0, 0), alpha=0.7)
                segmented_frame = cv2.putText(img=segmented_frame, text=f"fps: {self.fps}", org=(15, 30),
                                              fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                              color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow("Player Segmentation", segmented_frame)
                cv2.imshow("Background Model", self.background)

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

        if self.params['colorspace'] == "HSV":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Subtract the background
        fgMask = self.backSub.apply(frame)

        # Erosion and dilation to remove noise and fill gaps
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
        cv2.erode(fgMask, erode_kernel, fgMask, iterations=2)
        cv2.dilate(fgMask, dilate_kernel, fgMask, iterations=2)
        '''kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)'''

        # Filter out noise and smaller components, keeping the largest component
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            fgMask = np.zeros_like(fgMask)
            cv2.drawContours(fgMask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        return fgMask

    def get_frame(self):
        if not self.frame_buffer.empty():
            return self.frame_buffer.get()
        return None
    
    def calibration(self, total_frames=200, average_over=50):
        print("Calibrating background...")
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = self.capture.read()
            if not ret:
                continue
            frame_count += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Start averaging once the desired starting frame is reached
            if frame_count == total_frames - average_over + 1:
                self.background = gray_frame.astype("float")
            elif frame_count > total_frames - average_over:
                cv2.accumulateWeighted(gray_frame, self.background, 0.2)
            
        # Convert the accumulated background to uint8 for display and processing
        self.background = cv2.convertScaleAbs(self.background)
        
        # Display the background for verification
        cv2.imshow("Calibrated Background", self.background)
        cv2.waitKey(0)  # Wait for a key press to close the background display window
        cv2.destroyAllWindows()
