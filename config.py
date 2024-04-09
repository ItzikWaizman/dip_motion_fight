class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Buffer size for frame reading
        self.params['frame_buffer_size'] = 10

        # Optical flow parameters
        self.params['resize'] = True
        self.params['resize_factor'] = 1.0
        self.params['mag_thresh'] = 30
        self.params['mv_thresh'] = 0.1

        # bbox parameters
        self.params['bbox_height'] = 0.8
        self.params['bbox_width'] = 0.3
        self.params['act_box_width'] = 0.15
        self.params['spacing'] = 0.05

        # Delay time between consecutive press operations (kick/punch/jump/crouch)
        self.params['game_commands_delay_time'] = 0.3

        # Max number of pending game command request
        self.params['work_request_buffer_max_size'] = 100

        # Displays
        self.params['display_preprocess'] = True
        self.params['display_centroid'] = True
