from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
from robot_model import puma560
import re


def main():
    puma = puma560()
    print("please enter the joint variable (in degree):")
    for i in range(6):
        print(f"theta{i+1} {puma.joints[i]}", end=" ")
    print(":")
    rad = input()
    rad = re.split(r'\s+', rad)
    rad = [float(i) for i in rad]
    print(rad)
    puma.set_joint_angle(rad)
    puma.get_dh_trans()
    print("[n o a p]:\n", puma.dh_trans)
    psi,theta,phi = puma.get_euler()
    x,y,z = puma.get_pose()
    print(f"\noutput:\n{x} {y} {z} {psi} {theta} {phi}")

if __name__ == '__main__':
    main()