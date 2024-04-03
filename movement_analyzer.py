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
        self.display = params['display_centroid']
        self.params = params
        self.running = True
        self.initialized = False
        self._init_kalman_filter()

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
        self.kalman.measurementNoiseCov = np.array([[25, 0],
                                                    [0, 100]], np.float32)

    def estimate_centroid(self, frame):
        if np.sum(frame == 255) > self.params['mv_thresh'] * frame.size:
            cx, cy = utils.get_centroid(frame)
        else:
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
        if self.kalman.statePre is not None:
            self.kalman.predict()
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
        else:
            cx, cy = utils.get_centroid(frame)
            self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
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
            frame = self.frame_reader.get_flow()
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
            fgMask = self.frame_reader.get_mask()
            if fgMask is not None:
                cx, cy, vx, vy = self.estimate_centroid(fgMask)
                cx = int(cx)
                cy = int(cy)
                img_shape = fgMask.shape

                print(f"v_x = {vx:.2f}, v_y = {vy:.2f}")
                if self.display:
                    frame = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
                    centroid_frame = cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
                    centroid_frame = cv2.putText(img=centroid_frame, text=f"speed: {int(vx)},{int(vy)}", org=(15, 30),
                                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                                 color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                    bbox_width = int(self.params['bbox_width'] * img_shape[1])
                    bbox_height = int(self.params['bbox_height'] * img_shape[0])
                    body_box = (max(cx - bbox_width // 2, 0), max(cy - bbox_height // 2, 0))
                    bbox_width = min(bbox_width, img_shape[1] - 1 - body_box[0])
                    bbox_height = min(bbox_height, img_shape[0] - 1 - body_box[1])
                    cv2.rectangle(centroid_frame, pt1=body_box[:2],
                                  pt2=(body_box[0]+bbox_width, body_box[1]+bbox_height),
                                  color=(0, 255, 0),
                                  shift=0)
                    act_box_width = int(self.params['act_box_width'] * img_shape[1])
                    spacing = int(self.params['spacing'] * img_shape[1])
                    punch_box = (min(body_box[0] + bbox_width + spacing, img_shape[1] - 1), body_box[1])
                    kick_box = (min(body_box[0] + bbox_width + spacing, img_shape[1] - 1),
                                body_box[1] + 2 * bbox_height // 3)
                    act_box_width = min(act_box_width, img_shape[1] - 1 - body_box[0])

                    # Adjust kick box based on player being bent down or not

                    cv2.rectangle(centroid_frame, pt1=punch_box[:2],
                                  pt2=(punch_box[0] + act_box_width, punch_box[1] + bbox_height//3),
                                  color=(255, 0, 0),
                                  shift=0)

                    cv2.rectangle(centroid_frame, pt1=kick_box[:2],
                                  pt2=(kick_box[0] + act_box_width, kick_box[1] + bbox_height//3),
                                  color=(255, 0, 0),
                                  shift=0)

                    cv2.imshow("Centroid Frame", centroid_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                        break
            else:
                time.sleep(0.03)

        print("Track-Motion Terminated...")

