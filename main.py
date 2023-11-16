from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
from robot_model import puma560

def main():
    puma = puma560()
    puma.set_roate()
if __name__ == '__main__':
    main()