# TO-DO List

### Pending

### In Progress

### Done ✓

- [x] Add bounding box computation for regions of interest in the image.
- [x] Implement logic to transition between stand still/left/right motion as well as crouch/upright position.
- [x] Test the command API general functionality.
- [x] Add optical flow computation for use in instant action detection algorithm.
- [x] Test the command API with actual in game key presses.
- [x] Implement algorithm to identify instant actions based on optical flow in regions of interest.
- [x] Replace the noisy segmentation with head detection - we can use the location of the head as
an anchor for the player's location.
- [x] Organize the code of the movement analyzer:
  - [x] Add the thresholds to the config file.
  - [x] Organize the different actions in a dictionary.
- [x] Fine tune the parameters of the movement analyzer.
- [x] Support combos when related gestures are recorded with small delay between them.
