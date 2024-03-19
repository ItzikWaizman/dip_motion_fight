class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Buffer size for frame reading
        self.params['frame_buffer_size'] = 10

        # Background subtraction parameters
        self.params['colorspace'] = "RGB"
        self.params['threshold'] = 55
        self.params['history'] = 500
        self.params['max_var'] = 32
        self.params['dist2thresh'] = 800

        # Delay time between consecutive press operations (kick/punch/jump/crouch)
        self.params['game_commands_delay_time'] = 0.3

        # Max number of pending game command request
        self.params['work_request_buffer_max_size'] = 100

        # Displays
        self.params['display_preprocess'] = True
        self.params['display_centroid'] = True  # keep false until issue with display is resolved
