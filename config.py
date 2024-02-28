class Parameters(object):

    def __init__(self):
        self.params = dict()

        # Buffer size for frame reading
        self.params['frame_buffer_size'] = 10

        # Delay time between consecutive press operations (kick/punch/jump/crouch)
        self.params['game_commands_delay_time'] = 0.3

        # Max number of pending game command request
        self.params['work_request_buffer_max_size'] = 100