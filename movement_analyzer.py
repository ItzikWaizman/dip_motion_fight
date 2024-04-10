from threading import Thread
import cv2
import numpy as np
import utils
import time
import matplotlib.pyplot as plt


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
        self.mid_punch = False
        self.mid_kick = False
        self.mid_jump = False
        self.jump_time = time.time()
        self.punch_time = time.time()

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
        th = Thread(target=self.analyze_movements, daemon=True)
        th.start()
        return th
    
    def analyze_movements(self):
        while self.running:
            self.track_motion()
            self.analyze_gestures()

    def analyze_gestures(self):

        """
        Analyze press movements such as punch, kick, crouch or jump.
        """

        flow = self.frame_reader.get_flow()
        if flow is not None:
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
            cx = int(cx)
            cy = int(cy)

            body_box, punch_box, kick_box = utils.compute_roi(flow.shape,
                                                                centroid=(cx, cy),
                                                                bbox_shape=(self.width, self.height),
                                                                params=self.params,
                                                                plot=False)

            # Use optical flow to identify punch
            punch_box_flow = flow[punch_box[1]:punch_box[3]+1, punch_box[0]:punch_box[2]+1]
            mag, ang = cv2.cartToPolar(punch_box_flow[..., 0], punch_box_flow[..., 1])
            non_zero = mag > 1
            
            if np.sum(non_zero) > 0.05 * mag.size:
                if np.mean(mag[non_zero]) > 5:
                    if not self.mid_punch:
                        self.mid_punch = True
                        self.command_api.add_work_request("punch")
                        self.punch_time = time.time()
            if time.time() - self.punch_time > 0.3:
                self.mid_punch = False

            # Use optical flow to identify kick
            kick_box_flow = flow[kick_box[1]:kick_box[3]+1, kick_box[0]:kick_box[2]+1]
            mag, ang = cv2.cartToPolar(kick_box_flow[..., 0], kick_box_flow[..., 1])
            non_zero = mag > 1

            if np.sum(non_zero) > 0.05 * mag.size:
                if np.mean(mag[non_zero]) > 20:
                    if not self.mid_kick:
                        self.mid_kick = True
                        self.command_api.add_work_request("kick")
                if np.mean(mag) < 0.1:
                    self.mid_kick = False
            else:
                self.mid_kick = False

            # Use optical flow to identify jump
            body_box_height = body_box[3] - body_box[1]
            body_box_flow = flow[body_box[1]:body_box[1] +int(0.3*body_box_height) + 1, body_box[0]:body_box[2]+1]

            print(f"YAKOV: np.mean(biioukuidy_box_flow[..., 1]): {np.mean(body_box_flow[..., 1])}")
            if np.mean(body_box_flow[..., 1]) < -3:
                if not self.mid_jump:
                    self.mid_jump = True
                    self.command_api.add_work_request("jump")
                    self.jump_time = time.time()
            elif time.time() - self.jump_time > 1:
                self.mid_jump = False

            if self.display:
                visual = utils.optical_flow_visualization(flow)
                cv2.imshow("Optical Flow", visual)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                    self.running=False

        else:
            time.sleep(0.03)

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
        fgMask = self.frame_reader.get_mask()
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
                utils.compute_roi(fgMask.shape,
                                    centroid=(cx, cy),
                                    bbox_shape=(self.width, self.height),
                                    params=self.params,
                                    plot=True,
                                    target=centroid_frame)
                cv2.imshow("Centroid Frame", centroid_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                    self.running=False
        else:
            time.sleep(0.03)
