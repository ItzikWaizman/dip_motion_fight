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
        self.tracking = False
        self.skip_counter = 0
        self.display = self.params['display_tracking']
        self.running = True
        self._init_kalman_filter()
        # TODO: Tidy up the class - too many class variables.
        self.motion_state = "still"
        self.body_state = "upright"
        self.mid_punch = False
        self.mid_kick = False
        self.mid_jump = False
        self.jump_time = time.time()
        self.punch_time = time.time()
        self.kick_time = time.time()

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

    def track_head(self, x, y):
        if self.tracking:
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
            if utils.dist(x, y, cx, cy) > self.params['outlier_thresh']:
                if self.tracking:
                    self.skip_counter = self.skip_counter + 1
                if self.skip_counter >= self.params['skip_threshold']:
                    self.tracking = False
                    self.skip_counter = 0
            else:
                self.skip_counter = 0
                self.kalman.predict()
                self.kalman.correct(np.array([[x], [y]], np.float32))
        else:
            if self.skip_counter == 0:
                self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)

            cx, cy, _, _ = self.kalman.statePre.flatten().tolist()
            if utils.dist(x, y, cx, cy) <= self.params['outlier_thresh']:
                self.skip_counter = self.skip_counter - 1

            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            if self.skip_counter <= -10:
                self.skip_counter = 0
                self.tracking = True

        return self.kalman.statePost.flatten().tolist()

    def start_analysis(self):
        th = Thread(target=self.analyze_movements, daemon=True)
        th.start()
        return th
    
    def analyze_movements(self):
        while self.running:
            frame_analysis = self.frame_reader.get_frame_analysis()
            if frame_analysis is not None:
                (frame, opt_flow, head_circle) = frame_analysis
                self.track_motion(frame, head_circle)
                self.analyze_gestures(frame, opt_flow)

                if self.display:

                    if head_circle is not None:
                        (x, y, r) = head_circle
                        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), 3)

                    x, y, _, _ = self.kalman.statePost.flatten().tolist()
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), 3)

                    cv2.putText(img=frame, text=f"Tracking (counter = {self.skip_counter})", org=(15, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(0, 255, 0) if self.tracking else (0, 0, 255), thickness=2,
                                lineType=cv2.LINE_AA)

                    cv2.imshow("Head detection", frame)
                    # Break loop with 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                time.sleep(0.03)

        print("Analyze Movements Terminated...")

    def analyze_gestures(self, frame, opt_flow):

        """
        Analyze press movements such as punch, kick, crouch or jump.
        """

        if opt_flow is not None and self.tracking:
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
            cx = int(cx)
            cy = int(cy)
            width = int(frame.shape[1] * self.params['bbox_base_width'])
            height = int(frame.shape[0] * self.params['bbox_base_height'])
            body_box, punch_box, kick_box = utils.compute_roi(opt_flow.shape,
                                                              centroid=(cx, cy),
                                                              bbox_shape=(width, height),
                                                              params=self.params,
                                                              plot=self.display,
                                                              target=frame)

            act_box_width = punch_box[2] - punch_box[0]
            if act_box_width >= 0:
                # Use optical flow to identify punch
                punch_box_flow = opt_flow[punch_box[1]:punch_box[3]+1, punch_box[0]:punch_box[2]+1]
                mag, ang = cv2.cartToPolar(punch_box_flow[..., 0], punch_box_flow[..., 1])
                non_zero = mag > 1

                if np.sum(non_zero) > 0.2 * mag.size:
                    if np.mean(mag[non_zero]) > 10:
                        if not self.mid_punch:
                            self.mid_punch = True
                            self.command_api.add_work_request("punch")
                            self.punch_time = time.time()
                if time.time() - self.punch_time > 0.5:
                    self.mid_punch = False

                # Use optical flow to identify kick
                kick_box_flow = opt_flow[kick_box[1]:kick_box[3]+1, kick_box[0]:kick_box[2]+1]
                mag, ang = cv2.cartToPolar(kick_box_flow[..., 0], kick_box_flow[..., 1])
                non_zero = mag > 1

                if np.sum(non_zero) > 0.3 * mag.size:
                    if np.mean(mag[non_zero]) > 10:
                        if not self.mid_kick:
                            self.mid_kick = True
                            self.command_api.add_work_request("kick")
                            self.kick_time = time.time()
                if time.time() - self.kick_time > 0.5:
                    self.mid_kick = False
            else:
                self.mid_kick = False

            # Use optical flow to identify jump
            body_box_flow = opt_flow[body_box[1]:body_box[3]+1, body_box[0]:body_box[2]+1]

            if np.mean(body_box_flow[..., 1]) < -10:
                if not self.mid_jump:
                    self.mid_jump = True
                    self.command_api.add_work_request("jump")
                    self.jump_time = time.time()
            elif time.time() - self.jump_time > 1:
                self.mid_jump = False

    def track_motion(self, frame, head_circle):

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
        if head_circle is not None:
            (x, y, r) = head_circle
            self.track_head(x, y)
        elif self.tracking:
            self.skip_counter = self.skip_counter + 1

        if self.skip_counter >= self.params['skip_threshold']:
            self.tracking = False
            self.skip_counter = 0

        if self.tracking:
            # Analyze motion
            _, y_new, vx, _ = self.kalman.statePost.flatten().tolist()
            y_new = int(y_new)
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
            if y_new > self.params['crouch_thresh'] * frame.shape[0]:
                if self.body_state != "crouch":
                    self.command_api.add_work_request("crouch")
                    self.body_state = "crouch"

            else:
                if self.body_state != "upright":
                    self.command_api.add_work_request("upright")
                    self.body_state = "upright"
