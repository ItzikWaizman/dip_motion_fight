class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Frame reader parameters
        self.params['capture_device'] = 0  # External Camera
        self.params['frame_buffer_size'] = 10
        self.params['resize'] = 1  # Resize factor

        # Player tracking method (Haar / Circles)
        self.params['method'] = 'Haar'

        # Haar Cascade parameters
        self.params['Haar_scale'] = 1.1
        self.params['Haar_neighbors'] = 5
        self.params['Haar_min_face_size'] = (30, 30)

        # Bounding box parameters
        self.params['orientation'] = "right"
        # Bounding box (% of screen dimension)
        self.params['bbox_base_width'] = 0.25
        self.params['bbox_base_height'] = 0.8
        self.params['act_box_width'] = 0.15
        self.params['spacing'] = 0.2  # Space between action box and centroid

        # Kalman Filter Parameters
        self.params['obs_noise_var'] = [25, 25]
        self.params['outlier_thresh'] = 150
        self.params['skip_threshold'] = 10

        # Command API parameters
        self.params['game_commands_delay_time'] = 0.3
        self.params['work_request_buffer_max_size'] = 10
        self.params['motion_thresh'] = 10
        self.params['crouch_thresh'] = 0.5

        # Displays
        self.params['display_preprocess'] = True
        self.params['display_tracking'] = True
