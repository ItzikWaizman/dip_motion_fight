from config import Parameters
from frame_reader import FrameReader
from game_commands_api import CommandAPI
from movement_analyzer import MovementAnalyzer
from utils import WindowMgr
import pywintypes
import keyboard


def main():
    # Configurations dictionary
    params = Parameters().params

    # Create FramerReader
    frame_reader = FrameReader(params)

    # Create Game Command API
    command_api = CommandAPI(params)

    # Create MovementAnalyzer - Implements the actual algorithm to identify movements from FramerReader's data.
    movement_analyzer = MovementAnalyzer(frame_reader, command_api, params)

    # Create Window Manager
    wm = WindowMgr()

    # Transition into the game
    try:
        wm.find_window_wildcard(".*TEKKEN 7.*")
        wm.set_foreground()
    except pywintypes.error:
        print("Unable to find game window...")

    # Add pause button to alternate between game_mode and menue navigation 
    keyboard.add_hotkey('space', command_api.flip_game_mode)

    # Begin playing
    frThread = frame_reader.start_capture()
    movement_analyzer.start_analysis()

    # Exit code
    frThread.join()
    movement_analyzer.running = False
    command_api.running = False
    exit(0)


if __name__ == "__main__":
    main()
