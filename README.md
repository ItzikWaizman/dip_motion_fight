# DIP_MOTION_FIGHT

### Creators: Yakov Gusakov, Itzik Weizmann, Yoni Bartal and Yoav Meishar.
### Lecturer: Prof. Tammy Riklin Raviv.


## An exciting new way to play!

Our project uses classical digital image processing tools to allow a user to interface with the Tekken 7
fighting game, bringing the gaming experience to life.

The entire project is based on classical image processing methods. The player tracking is done using a Haar 
cascade classifier, whose observations are fed into a Kalman filter for smooth motion tracking.
Punches, kicks and jumps are detected using optical flow.

This is our final project for the "Digital Image Processing" course at BGU.

<img src="./figures/Figure1.jpg" width="907" alt=""/>

## Installation
Clone the repository, and use the provided `requirements.txt` file and install the required packages:
```
pip install -r requirements.txt
```

If you're running on a laptop and use an external camera, set the "capture device" to 1. Otherwise, set it to 0.

Depending on the orientation of your camera, you might need to edit the "orientation" parameter in the config file
according to the position of the camera (to your left or to your right) when looking at the screen.

In theory this program can work with any fighting game, but you'll need to adapt the controls in
`game_commands_api`.

## Activation
Run the `dip_motion_fight.py` script when the game is already running. It should transition to the game 
window automatically. Otherwise, do so manually after starting the game. Choose a game format and character manually.

To activate or deactivate the movement analysis and command API, press the `space` button on your keyboard. 
It's recommended to choose a character and activate just before the battle is about to start.

The program prints the executed commands to the terminal for easy evaluation even without the game running.
