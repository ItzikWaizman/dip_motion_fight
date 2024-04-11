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
        self.params = params
        self.frame_analysis_buffer = queue.Queue(maxsize=self.params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(self.params['capture_device'])
        self.prev_gray = None
        self.fps = 0
        self.display = self.params['display_preprocess']
        if self.params['method'] == 'Haar':
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        else:
            self.face_cascade = None

    def start_capture(self):
        th = Thread(target=self.read_frames, daemon=True)
        th.start()
        return th

    def read_frames(self):
        i = 0

        # Initialize previous frame
        ret, frame = self.capture.read()
        if not ret:
            self.capture.release()
            print("Error capturing frames...")

        if self.params['resize'] != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.params['resize'], fy=self.params['resize'])

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start_time = time.time()

        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            if self.params['orientation'] == "right":
                frame = cv2.flip(frame, 1)

            # Resize if needed
            if self.params['resize'] != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=self.params['resize'], fy=self.params['resize'])

            # Compute optical flow
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.max(cv2.absdiff(self.prev_gray, gray)) != 0:
                opt_flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, flow=None, pyr_scale=0.5, levels=3,
                                                        winsize=15, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)
            else:
                opt_flow = None

            # Locate player's head
            head_circle = self.locate_head(gray)

            # Save gray frame
            self.prev_gray = gray

            # Enqueue frame analysis
            if self.frame_analysis_buffer.full():
                self.frame_analysis_buffer.get()
            self.frame_analysis_buffer.put((frame, opt_flow, head_circle))

            # Display segmentation and background
            if self.display:
                if opt_flow is not None:
                    visual_optical_flow = utils.optical_flow_visualization(opt_flow)
                else:
                    visual_optical_flow = np.zeros_like(frame)

                # Add fps display
                cv2.putText(img=visual_optical_flow, text=f"fps: {self.fps}", org=(15, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

                cv2.imshow("Optical Flow", visual_optical_flow)

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

    def locate_head(self, frame):

        faces = self.face_cascade.detectMultiScale(frame,
                                                   scaleFactor=self.params['Haar_scale'],
                                                   minNeighbors=self.params['Haar_neighbors'],
                                                   minSize=self.params['Haar_min_face_size'])
        if len(faces) > 0:
            x, y, h, w = faces[0]
            head_circle = (int(x + h / 2), int(y + w / 2), int(h / 2))
            return head_circle
        else:
            return None

    def get_frame_analysis(self):
        if not self.frame_analysis_buffer.empty():
            return self.frame_analysis_buffer.get()
        return None
