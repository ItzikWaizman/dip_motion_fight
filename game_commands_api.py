import pydirectinput as input
import random
from threading import Thread
from queue import Queue
import time
import utils


class CommandAPI:

    """
    Responsibilities: Translate movement commands that are identified by a MovementAnalyzer object 
    into real game commands.
    Implementation: We maintain a work_request_queue. This queue will be filled by MovementAnalyzer threads for 
    press actions and motion tracking. 
    A dedicated thread will iterate through the requests in the buffer and transfer them to the corresponding handlers.
    Motion requests are handled with an FSM, including 'Stand', 'Left' and 'Right' states.
    Press actions are directly executed and can be either punch, kick, jump or crouch.
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
        """Flip the game_mode flag. Commands can be sent to the game only when game_mode is True"""
        self.game_mode = not self.game_mode
        self.stop()
        print(f"game_mode = {self.game_mode}")

    # API to MovementAnalyzer threads
    def add_work_request(self, command):
        """Add a work request to the queue"""
        if self.game_mode:
            self.work_request_queue.put(command)

    def process_work_requests(self):

        """
        Process work requests from the queue.
        """

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

        self.stop()
        print("Command API terminated...")

    def handle_motion(self, command):

        """
        Handle a motion command.

        :param command: The motion command to handle.
        """

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
        """
        Execute an instant action command.

        :param command:  The instant action command to execute.
        """
        if command in ["punch", "kick"]:
            if not self.work_request_queue.empty():
                if self.work_request_queue.queue[0] in ["punch", "kick"]:
                    self.work_request_queue.get()
                    command_index = random.choice([0, 1])
                    action1 = self.instant_actions_map["punch"][command_index]
                    action2 = self.instant_actions_map["kick"][command_index]
                    utils.press_multiple_button(action1 + action2)
                    print(f"perform combo punch({action1}) + kick({action2})")
            else:
                action = random.choice(self.instant_actions_map[command])
                input.press(action)
                print(f"perform {command}")
        else:
            input.keyDown("up")
            time.sleep(0.2)
            input.keyUp("up")
            print(f"perform {command}")

    def stop(self):
        """Signal to stop the worker thread and ensure all keys are released"""
        if self.motion_state != "still":
            input.keyUp(self.motion_state)
            print(f"release {self.motion_state}")
        if self.body_state != "upright":
            input.keyUp("down")
            print("release down")
