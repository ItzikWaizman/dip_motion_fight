# TO-DO List

### Pending

- [ ] Integrate Loggers to debug and keep track about the different states of each thread.
- [ ] Propose different features to enhance user's experience, for example:
  - [ ] Voice detection to navigate through game's menu
- [ ] Try to distinguish between left and right kicks/punches.


### In Progress

- [ ] Test bounding box parameters.
- [ ] Implement algorithm to identify instant actions based on optical flow in regions of interest.
- [ ] Test the command API with actual in game key presses.


### Done ✓

- [x] Add bounding box computation for regions of interest in the image.
- [x] Implement logic to transition between stand still/left/right motion as well as crouch/upright position.
- [x] Test the command API general functionality.
- [x] Add optical flow computation for use in instant action detection algorithm.