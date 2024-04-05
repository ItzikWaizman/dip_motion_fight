class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Buffer size for frame reading
        self.params['frame_buffer_size'] = 10

        # Background subtraction parameters
        self.params['colorspace'] = "RGB"
        self.params['history'] = 200
        self.params['max_var'] = 32
        self.params['dist2thresh'] = 800
        self.params['kNN_Samples'] = 4

        # Bounding box parameters
        self.params['orientation'] = "right"
        self.params["dynamic_bbox"] = False
        self.params['bbox_base_width'] = 0.25
        self.params['bbox_base_height'] = 0.7
        self.params['act_box_width'] = 0.15
        self.params['spacing'] = 0.1

        # Kalman Filter Parameters
        self.params['obs_noise_var'] = [25, 100]
        self.params['mask_upper_thresh'] = 0.3
        self.params['mask_lower_thresh'] = 0.14

        # Delay time between consecutive press operations (kick/punch/jump/crouch)
        self.params['game_commands_delay_time'] = 0.3

        # Max number of pending game command request
        self.params['work_request_buffer_max_size'] = 100

        # Displays
        self.params['display_preprocess'] = True
        self.params['display_centroid'] = True
