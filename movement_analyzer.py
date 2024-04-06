from threading import Thread
import cv2
import numpy as np
import utils
import time


class MovementAnalyzer:

    """
    Responsibilities: Analyze body movements of the player, as well as keep tracking of player's location
    and post a suitable work request on a dedicated buffer of a CommandAPI object, to be executed and 
    translated into real game action.
    """

    def __init__(self, frame_reader, command_api, params):
        self.frame_reader = frame_reader
        self.command_api = command_api
        self.params = params
        self.display = self.params['display_centroid']
        self.initialized = False
        self.running = True
        self._init_kalman_filter()
        self.width = 0
        self.height = 0
        self.motion_state = "still"
        self.body_state = "upright"

    def _init_kalman_filter(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        self.kalman.measurementNoiseCov = np.array(np.diag(self.params['obs_noise_var']), np.float32)

    def estimate_centroid(self, frame):
        if self.params['mask_lower_thresh'] * frame.size < \
                np.sum(frame == 255) < self.params['mask_upper_thresh'] * frame.size:
            if self.params['dynamic_bbox']:
                cx, cy, self.width, self.height = utils.get_centroid(frame)
            else:
                cx, cy, _, _ = utils.get_centroid(frame)
        else:
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
        if self.initialized:
            self.kalman.predict()
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
        else:
            self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.width = int(self.params['bbox_base_width'] * frame.shape[1])
            self.height = int(self.params['bbox_base_height'] * frame.shape[0])
            self.initialized = True

        return self.kalman.statePost.flatten().tolist()

    def start_analysis(self):
        th1 = Thread(target=self.track_motion, daemon=True)
        th1.start()

        while not self.initialized:
            time.sleep(1)
        # th2 =  Thread(target=self.analyze_gestures, daemon=True)
        # th2.start()
        return th1

    def analyze_gestures(self):

        """
        Analyze press movements such as punch, kick, crouch or jump.
        Can be implemented using bound box to identify punch/kick, and center of mass tracking
        to identify crouch/jump.
        """

        while True:
            frame = self.frame_reader.get_frame()
            if frame is not None:
                # TODO: Implement analyze logic
                return None

    def track_motion(self):

        """
        We implement a final state machine, including three states:
        1. 'Stand'
        2. 'Left'
        3. 'Right'
        The movements should be continuous, so initiating movement to a specific direction includes sending a movement
        command to the CommandAPI with the corresponding direction, and to stop movement in this direction we can either
        send a movement command with the opposite direction or send a 'Stand' command.
        This can be implemented using Kalman filter or center of mass tracking.        
        """

        while self.running:
            fgMask = self.frame_reader.get_frame()
            if fgMask is not None:
                cx, cy, vx, vy = self.estimate_centroid(fgMask)
                cx = int(cx)
                cy = int(cy)

                # Analyze motion
                if abs(vx) > self.params['motion_thresh']:
                    if vx > self.params['motion_thresh']:
                        if self.motion_state != "left":
                            self.command_api.add_work_request("left")
                            self.motion_state = "left"
                    else:
                        if self.motion_state != "right":
                            self.command_api.add_work_request("right")
                            self.motion_state = "right"
                else:
                    if self.motion_state != "still":
                        self.command_api.add_work_request("still")
                        self.motion_state = "still"

                # Analyze body position
                if cy > self.params['crouch_thresh'] * fgMask.shape[0]:
                    if self.body_state != "crouch":
                        self.command_api.add_work_request("crouch")
                        self.body_state = "crouch"

                else:
                    if self.body_state != "upright":
                        self.command_api.add_work_request("upright")
                        self.body_state = "upright"

                # Display
                if self.display:
                    frame = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
                    centroid_frame = cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
                    centroid_frame = cv2.putText(img=centroid_frame, text=f"speed: {int(vx)},{int(vy)}", org=(15, 30),
                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                                 color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    utils.compute_roi(centroid_frame,
                                      centroid=(cx, cy),
                                      dimensions=(self.width, self.height),
                                      params=self.params)
                    cv2.imshow("Centroid Frame", centroid_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                        break
            else:
                time.sleep(0.03)

        print("Track-Motion Terminated...")
