from threading import Thread
import cv2
import numpy as np
import utils


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
        self.kalman.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32)

    def estimate_centroid(self, frame):
        cx, cy = utils.get_centroid(frame)
        if self.kalman.statePre is not None:
            self.kalman.predict()
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
        else:
            cx, cy = utils.get_centroid(frame)
            self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

        return self.kalman.statePost.flatten().tolist()

    def start_analysis(self):
        # TODO: Need to implement reading of a frame and spread it to both threads.
        # Thread(target=self.analyze_gestures, daemon=True).start()
        Thread(target=self.track_motion, daemon=False).start()

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

        while True:
            frame = self.frame_reader.get_frame()
            if frame is not None:
                cx, cy, vx, vy = self.estimate_centroid(frame)
                cx = int(cx)
                cy = int(cy)
                print(f"speed: {vx},{vy}")
                if self.display:
                    pass
                    # TODO: Figure out how to display the centroid - can't plot a red dot on a binary frame.
                    centroid_frame = cv2.circle(frame, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.imshow("Centroid Frame", centroid_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop with 'q' key
                        break

        print("Track-Motion Terminated...")
