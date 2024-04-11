# TO-DO List

### Pending

- [ ] Propose different features to enhance user's experience, for example:
  - [ ] Voice detection to navigate through game's menu
  - [ ] Method to flip the controls if the characters change places in game.
- [ ] Try to distinguish between left and right kicks/punches.
- [ ] Support combos when related gestures are recorded with small delay between them.


### In Progress

- [ ] Organize the code of the movement analyzer:
  - [ ] Add the thresholds to the config file.
  - [ ] Organize the different actions in a dictionary. 
- [ ] Fine tune the parameters of the movement analyzer.

### Done ✓

- [x] Add bounding box computation for regions of interest in the image.
- [x] Implement logic to transition between stand still/left/right motion as well as crouch/upright position.
- [x] Test the command API general functionality.
- [x] Add optical flow computation for use in instant action detection algorithm.
- [x] Test the command API with actual in game key presses.
- [x] Implement algorithm to identify instant actions based on optical flow in regions of interest.
- [x] Replace the noisy segmentation with head detection - we can use the location of the head as
an anchor for the player's location.