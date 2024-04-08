class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Frame reader parameters
        self.params['capture_device'] = 1 # External Camera
        self.params['frame_buffer_size'] = 10
        self.params['resize'] = 0.5 # Resize factor

        # Background subtraction parameters
        self.params['colorspace'] = "RGB"
        self.params['history'] = 200
        self.params['max_var'] = 32
        self.params['dist2thresh'] = 400
        self.params['kNN_Samples'] = 4

        # Bounding box parameters
        self.params['orientation'] = "left"
        self.params["dynamic_bbox"] = False
        # Bouding box (% of screen dimension)
        self.params['bbox_base_width'] = 0.25
        self.params['bbox_base_height'] = 0.8
        self.params['act_box_width'] = 0.15
        self.params['spacing'] = 0.2 # Space between action box and centroid

        # Kalman Filter Parameters
        self.params['obs_noise_var'] = [5, 5]
        # Mask size threshold for movement tracking (% of image)
        self.params['mask_upper_thresh'] = 0.3
        self.params['mask_lower_thresh'] = 0.1

        # Command API parameters
        self.params['game_commands_delay_time'] = 0.3
        self.params['work_request_buffer_max_size'] = 100
        self.params['motion_thresh'] = 5
        self.params['crouch_thresh'] = 0.6

        # Displays
        self.params['display_preprocess'] = True
        self.params['display_movement'] = True
