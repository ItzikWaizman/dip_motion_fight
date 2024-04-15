class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Frame reader parameters
        self.params['capture_device'] = 0  # External Camera
        self.params['frame_buffer_size'] = 10
        self.params['resize'] = 0.5  # Resize factor

        # Player tracking method (Haar / Circles)
        self.params['method'] = 'Haar'

        # Haar Cascade parameters
        self.params['Haar_scale'] = 1.1
        self.params['Haar_neighbors'] = 5
        self.params['Haar_min_face_size'] = (int(30 * self.params['resize']), int(30 * self.params['resize']))

        # Bounding box parameters
        self.params['orientation'] = "right"
        # Bounding box (% of screen dimension)
        self.params['bbox_base_width'] = 0.25
        self.params['bbox_base_height'] = 0.8
        self.params['act_box_width'] = 0.15
        self.params['spacing'] = 0.2  # Space between action box and centroid

        # Instant action parameters
        self.params['time_between_actions'] = 0.5
        self.params['time_between_jumps'] = 1
        self.params['opt_flow_threshold'] = 10 * self.params['resize']
        self.params['opt_flow_punch_presence'] = 0.2
        self.params['opt_flow_kick_presence'] = 0.3

        # Kalman Filter & tracking parameters
        self.params['obs_noise_var'] = [int(50 * self.params['resize']), int(50 * self.params['resize'])]
        self.params['outlier_thresh'] = 150 * self.params['resize']
        self.params['skip_threshold'] = 10
        self.params['motion_thresh'] = 7 * self.params['resize']
        self.params['crouch_thresh'] = 0.5

        # Command API parameters
        self.params['game_commands_delay_time'] = 0.03
        self.params['work_request_buffer_max_size'] = 10

        # Display parameters
        self.params['display_preprocess'] = True
        self.params['display_tracking'] = True
