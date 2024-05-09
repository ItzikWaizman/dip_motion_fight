from threading import Thread
import cv2
import numpy as np
import utils
import time
import pickle


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
        self.state_dict = {'motion': "still", 'body': "upright"}
        self.mid_action_dict = {'punch': False, 'kick': False, 'jump': False}
        self.action_times_dict = {'punch': 0.0, 'kick': 0.0, 'jump': 0.0}

    def _init_kalman_filter(self):
        """Initializes the kalman filter."""
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

        """
        Responsible for tracking the location of the player's head.
        :param x: x coordinate of the player's head observation.
        :param y: y coordinate of the player's head observation.
        :return: Filtered Kalman state vector (x, y, vx, vy).
        """

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
        """Creates a movement analysis thread"""
        th = Thread(target=self.analyze_movements, daemon=True)
        th.start()
        return th
    
    def analyze_movements(self):

        """
        Analyzes the frame for movement and gestures using the data from the frame reader.
        """

        while self.running:
            frame_analysis = self.frame_reader.get_frame_analysis()
            if frame_analysis is not None:

                # Get frame, optical flow and head location from the queue.
                (frame, opt_flow, head_circle) = frame_analysis
                # Track motion
                self.track_motion(frame, head_circle)
                # Analyze gestures
                self.analyze_gestures(frame, opt_flow)

                # Display player tracking
                if self.display:

                    # Resize for visualization
                    frame = cv2.resize(frame, (0, 0), fx=1/self.params['resize'], fy=1/self.params['resize'])

                    # Draw the player's head observed location and Kalman estimation.
                    if head_circle is not None:
                        (x, y, r) = head_circle
                        x = int(x / self.params['resize'])
                        y = int(y / self.params['resize'])
                        r = int(r / self.params['resize'])
                        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), 3)

                    x, y, _, _ = self.kalman.statePost.flatten().tolist()
                    cv2.circle(frame, (int(x/self.params['resize']), int(y/self.params['resize'])),
                               2, (255, 0, 0), 3)

                    # Display tracking counter
                    cv2.putText(img=frame, text=f"Tracking (skips = {self.skip_counter})", org=(15, 30),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(0, 255, 0) if self.tracking else (0, 0, 255), thickness=2,
                                lineType=cv2.LINE_AA)

                    cv2.imshow("Player Tracking", frame)

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
            # Use the Kalman estimation to compute regions of interest
            cx, cy, _, _ = self.kalman.statePost.flatten().tolist()
            cx = int(cx)
            cy = int(cy)
            width = int(frame.shape[1] * self.params['bbox_base_width'])
            height = int(frame.shape[0] * self.params['bbox_base_height'])
            bounding_boxes = utils.compute_roi(opt_flow.shape,
                                               centroid=(cx, cy),
                                               bbox_shape=(width, height),
                                               params=self.params,
                                               plot=self.display,
                                               target=frame)

            act_box_width = bounding_boxes['punch'][2] - bounding_boxes['punch'][0]
            if act_box_width >= 0:
                # Use optical flow to identify punch/kick
                for action in ['punch', 'kick']:

                    box_flow = opt_flow[bounding_boxes[action][1]:bounding_boxes[action][3]+1,
                                        bounding_boxes[action][0]:bounding_boxes[action][2]+1]
                    mag, ang = cv2.cartToPolar(box_flow[..., 0], box_flow[..., 1])
                    non_zero = mag > 1

                    if np.sum(non_zero) > self.params['opt_flow_punch_presence'] * mag.size:
                        if np.mean(mag[non_zero]) > self.params['opt_flow_threshold']:
                            if not self.mid_action_dict[action]:
                                self.mid_action_dict[action] = True
                                self.command_api.add_work_request(action)
                                self.action_times_dict[action] = time.time()
                    if time.time() - self.action_times_dict[action] > self.params['time_between_actions'][action]:
                        self.mid_action_dict[action] = False

            # Use optical flow to identify jump
            body_box_flow = opt_flow[bounding_boxes['body'][1]:bounding_boxes['body'][3]+1,
                                     bounding_boxes['body'][0]:bounding_boxes['body'][2]+1]

            if np.mean(body_box_flow[..., 1]) < - self.params['opt_flow_threshold']:
                if not self.mid_action_dict['jump']:
                    self.mid_action_dict['jump'] = True
                    self.command_api.add_work_request("jump")
                    self.action_times_dict['jump'] = time.time()
            elif time.time() - self.action_times_dict['jump'] > self.params['time_between_jumps']:
                self.mid_action_dict['jump'] = False

    def track_motion(self, frame, head_circle):

        """
        Implements a final state machine, including three states:
        1. 'Stand'
        2. 'Left'
        3. 'Right'
        The movements should be continuous, so initiating movement to a specific direction includes sending a movement
        command to the CommandAPI with the corresponding direction, and to stop movement in this direction we can either
        send a movement command with the opposite direction or send a 'Stand' command.
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
                    if self.state_dict['motion'] != "left":
                        self.command_api.add_work_request("left")
                        self.state_dict['motion'] = "left"
                else:
                    if self.state_dict['motion'] != "right":
                        self.command_api.add_work_request("right")
                        self.state_dict['motion'] = "right"
            else:
                if self.state_dict['motion'] != "still":
                    self.command_api.add_work_request("still")
                    self.state_dict['motion'] = "still"

            # Analyze body position
            if y_new > self.params['crouch_thresh'] * frame.shape[0]:
                if self.state_dict['body'] != "crouch":
                    self.command_api.add_work_request("crouch")
                    self.state_dict['body'] = "crouch"

            else:
                if self.state_dict['body'] != "upright":
                    self.command_api.add_work_request("upright")
                    self.state_dict['body'] = "upright"
