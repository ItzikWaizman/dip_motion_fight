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
        self.work_request_queue = Queue(maxsize=params['work_request_buffer_max_size'])
        self.delay_time = params['game_commands_delay_time']
        self.running = True
        self.motion_state = "still"
        self.body_state = "upright"
        
        # Handlers for FSM actions
        self.motion_state_actions = ["still", "left", "right", "crouch", "upright"]
        
        # Mapping for instant press actions
        self.instant_actions_map = {
            "punch": ["u", "i"],
            "kick": ["j", "k"],
            "jump": "up",
        }
        
        # Start the thread that processes work requests
        self.worker_thread = Thread(target=self.process_work_requests, daemon=False)
        self.worker_thread.start()

    # API to MovementAnalyzer threads
    def add_work_request(self, command):
        self.work_request_queue.put(command)

    def process_work_requests(self):
        while self.running:
            if not self.work_request_queue.empty():
                command = self.work_request_queue.get()
                if command in self.motion_state_actions:
                    self.handle_motion(command)
                elif command in self.instant_actions_map:
                    self.execute_instant_action(command)
                else:
                    print("Error, unrecognized command...")

                # Wait configurable delay time between consecutive operations.
                time.sleep(self.delay_time)
            else:
                time.sleep(self.delay_time)

    def handle_motion(self, command):
        if command in ["still", "left", "right"]:
            if command != self.motion_state and self.motion_state != "still":
                print(f"release {self.motion_state}")  # input.keyUp(self.motion_state)
            if command == "still":
                self.motion_state = "still"
            else:
                self.motion_state = command
                print(f"press {self.motion_state}")  # input.keyDown(self.motion_state)
        else:
            if command != self.body_state and self.body_state != "upright":
                print(f"release down")  # input.keyUp("down")
            if command == "upright":
                self.body_state = "upright"
            else:
                self.body_state = "crouch"
                print(f"press down")  # input.keyDown("down")

    def execute_instant_action(self, command):
        # Execute press actions with a delay
        if command in ["punch", "kick"]:
            action = random.choice(self.instant_actions_map[command])
        else:
            action = self.instant_actions_map[command]
        print(f"perform {command}")  # input.press(action)

    def stop(self):
        # Signal to stop the worker thread and ensure all keys are released
        self.running = False
        if self.motion_state != "still":
            print(f"release {self.motion_state}")  # input.keyUp(self.motion_state)
        if self.body_state != "upright":
            print("release down")  # input.keyUp("down")
