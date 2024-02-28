import pydirectinput as input
import random
from threading import Thread
from queue import Queue
import time

class CommandAPI:

    """
    Responsibilities: Translate movement commands that are identified by a MovementAnalyzer object 
    into real game commands.
    Implementation: We maintain a work_request_queue. This queue will be filled by MovementAnalyzer threads for 
    press actions and motion tracking. 
    A dedicated thread will iterate through the requests in the buffer and transfer them to the corresponding handlers.
    Motion requests are handled with a FSM, including 'Stand', 'Left' and 'Right' states.
    Press actions are directly executed and can be either punch, kick, jump or crouch.
    TODO: Support Combos.
    """
        
    def __init__(self, params):
        self.current_state = 'Stand'
        self.work_request_queue = Queue(maxsize=params['work_request_buffer_max_size'])
        self.delay_time = params['game_commands_delay_time']
        self.running = True
        
        # Handlers for FSM actions
        self.motion_state_actions = {
            'Stand': self.handle_standing,
            'Left': self.handle_left,
            'Right': self.handle_right,
        }
        
        # Mapping for instant press actions
        self.instant_actions_map = {
            "punch": ["u", "i"],
            "kick": ["j", "k"],
            "jump": "up",
            "crouch": "down",
        }
        
        # Start the thread that processes work requests
        self.worker_thread = Thread(target=self.process_work_requests, daemon=True)
        self.worker_thread.start()

    # API to MovementAnalyzer threads
    def add_work_request(self, command):
        self.work_request_queue.put(command)

    def process_work_requests(self):
        while self.running:
            if not self.work_request_queue.empty():
                command = self.work_request_queue.get()
                if command in self.motion_state_actions:
                    self.motion_state_actions[command](command)
                elif command in self.instant_actions_map:
                    self.execute_instant_action(command)

                # Wait configurable delay time between consecutive operations.
                time.sleep(self.delay_time)

    def handle_standing(self, command):
        if command == 'Left':
            input.keyDown('left')
            self.current_state = 'Left'
        elif command == 'Right':
            input.keyDown('right')
            self.current_state = 'Right'

    def handle_left(self, command):
        if command == 'Stand':
            input.keyUp('left')
            self.current_state = 'Stand'
        elif command == 'Right':
            input.keyUp('left')
            input.keyDown('right')
            self.current_state = 'Right'

    def handle_right(self, command):
        if command == 'Stand':
            input.keyUp('right')
            self.current_state = 'Stand'
        elif command == 'Left':
            input.keyUp('right')
            input.keyDown('left')
            self.current_state = 'Left'

    def execute_instant_action(self, command):
        # Execute press actions with a delay
        if command in ["punch", "kick"]:
            action = random.choice(self.instant_actions_map[command])
        else:
            action = self.instant_actions_map[command]
        input.press(action)

    def stop(self):
        # Signal to stop the worker thread and ensure all keys are released
        self.running = False
        if self.current_state == 'Left':
            input.keyUp('left')
        elif self.current_state == 'Right':
            input.keyUp('right')
