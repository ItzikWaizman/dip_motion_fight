from threading import Thread

class MovementAnalyzer:
    
    """
    Responsibilities: Analyze body movements of the player, as well as keep tracking of player's location
    and post a suitable work request on a dedicated buffer of a CommandAPI object, to be executed and 
    translated into real game action.
    """

    def __init__(self, frame_reader, command_api):
        self.frame_reader = frame_reader
        self.command_api = command_api

    def start_analysis(self):
        #TODO: Need to implement reading of a frame and spread it to both threads.
        Thread(target=self.analyze_movement, daemon=True).start()
        Thread(target=self.track_motion, daemon=True).start()

    def analyze_movement(self):

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
                # TODO: Implement motion tracking logic
                return None