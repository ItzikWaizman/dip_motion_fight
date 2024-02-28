from config import Parameters
from frame_reader import FrameReader
from game_commands_api import CommandAPI
from movement_analyzer import MovementAnalyzer

def main():

    # Configurations dictionary
    params = Parameters().params

    # Create FramerReader
    frame_reader = FrameReader(params)

    # Create Game Command API
    command_api = CommandAPI(params)

    # Create MovementAnalyzer - Implements the actual algorithm to identify movements from FramerReader's data.
    movement_analyzer = MovementAnalyzer(frame_reader, command_api)

    frame_reader.start_capture()
    movement_analyzer.start_analysis()


if __name__ == "__main__":
    main()
