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
    Motion requests are handled with an FSM, including 'Stand', 'Left' and 'Right' states.
    Press actions are directly executed and can be either punch, kick, jump or crouch.
    TODO: Support Combos.
    """
        
    def __init__(self, params):
        self.work_request_queue = Queue(maxsize=params['work_request_buffer_max_size'])
        self.delay_time = params['game_commands_delay_time']
        self.running = True
        self.motion_state = "still"
        self.body_state = "upright"
        self.game_mode = False
        
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

    def flip_game_mode(self):
        self.game_mode = not self.game_mode
        print(f"game_mode = {self.game_mode}")

    # API to MovementAnalyzer threads
    def add_work_request(self, command):
        if self.game_mode:
            self.work_request_queue.put(command)

    def process_work_requests(self):
        while self.running:
            if not self.work_request_queue.empty():
                print(self.work_request_queue.queue)
                command = self.work_request_queue.get()
                if command in self.motion_state_actions:
                    self.handle_motion(command)
                elif command in self.instant_actions_map:
                    self.execute_instant_action(command)
                else:
                    print("Error, unrecognized command...")
            else:
                time.sleep(self.delay_time)

    def handle_motion(self, command):
        if command in ["still", "left", "right"]:
            if command != self.motion_state and self.motion_state != "still":
                input.keyUp(self.motion_state)
                print(f"release {self.motion_state}")
            if command == "still":
                self.motion_state = "still"
            else:
                self.motion_state = command
                input.keyDown(self.motion_state)
                print(f"press {self.motion_state}")
        else:
            if command != self.body_state and self.body_state != "upright":
                input.keyUp("down")
                print(f"release down")
            if command == "upright":
                self.body_state = "upright"
            else:
                self.body_state = "crouch"
                input.keyDown("down")
                print(f"press down")

    def execute_instant_action(self, command):
        # Execute press actions with a delay
        if command in ["punch", "kick"]:
            action = random.choice(self.instant_actions_map[command])
            input.press(action)
            print(f"perform {command}")
        else:
            input.keyDown("up")
            time.sleep(0.2)
            input.keyUp("up")
            print(f"perform {command}")

    def stop(self):
        # Signal to stop the worker thread and ensure all keys are released
        self.running = False
        if self.motion_state != "still":
            input.keyUp(self.motion_state)
            print(f"release {self.motion_state}")
        if self.body_state != "upright":
            input.keyUp("down")
            print("release down")
