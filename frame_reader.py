import queue
import cv2
from threading import Thread

class FrameReader:

    """
    Responsibilities: Read frames from the camera and perform initial processing - mainly player segmentation.
    The processed frames will be stored in a queue which will serve the consumer for movement analysis.
    """

    def __init__(self, params):
        self.frame_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(0)  # Assuming default camera

    def start_capture(self):
        Thread(target=self.read_frames, daemon=True).start()

    def read_frames(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            segmented_frame = self.segment_player(frame)
            if not self.frame_buffer.full():
                self.frame_buffer.put(segmented_frame)

    def segment_player(self, frame):
        # TODO: Implement player segmentation. 
        return frame  # This should be the segmented frame

    def get_frame(self):
        if not self.frame_buffer.empty():
            return self.frame_buffer.get()
        return None
    
    def calibration(self):
        # TODO: Implement calibration stage. Might use to save the background and subtruct it from next frames.
        return None