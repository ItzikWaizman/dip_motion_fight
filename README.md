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

## Installation
Clone the repository, and use the provided `requirements.txt` file and install the required packages:
```
pip install -r requirements.txt
```

In theory this program can work with any fighting game, but you'll need to adapt the controls in
`game_commands_api`.

<img src="./figures/Figure1.jpg" width="907" alt=""/>
