import queue
import cv2
import numpy as np
from threading import Thread
from movement_analyzer import MovementAnalyzer


class FrameReader:

    """
    Responsibilities: Read frames from the camera and perform initial processing - mainly player segmentation.
    The processed frames will be stored in a queue which will serve the consumer for movement analysis.
    """

    def __init__(self, params):
        self.frame_buffer = queue.Queue(maxsize=params['frame_buffer_size'])
        self.capture = cv2.VideoCapture(0)  # Assuming default camera
        self.background = None
        self.display = params['display_preprocess']

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
            if self.display:
                cv2.imshow("Segmented Frame", segmented_frame)  # Display the segmented frame
                # cv2.imshow("Original Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                break
        self.capture.release()
        cv2.destroyAllWindows()
        print("Frame Reader Terminated...")

    def segment_player(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Subtract the background
        foreground_mask = cv2.absdiff(self.background, gray_frame)
        # Apply a binary threshold to get a binary mask
        _, binary_mask = cv2.threshold(foreground_mask, 50, 255, cv2.THRESH_BINARY)

        # Filter out noise and smaller components, keeping the largest component
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            binary_mask = np.zeros_like(binary_mask)
            cv2.drawContours(binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Apply median filter
        filtered_mask = cv2.medianBlur(binary_mask, 5)

        return filtered_mask

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
