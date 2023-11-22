import numpy as np
from math import pi
from robot_model import puma560
import re


def main():
    puma = puma560()
    auto = input("auto input y/n:")
    print("please enter the joint variable (in degree):")
    for i in range(6):
        print(f"theta{i+1} {puma.joints[i]}", end=" ")
    print(":")
    
    rad = [20,20,20,20,20,20]
    if auto == "n":
        rad = input()
        rad = re.split(r'\s+', rad)
        rad = [float(i) for i in rad]
    print(rad)
    puma.set_joints_angle(rad)
    print("[n o a p]:\n", puma.get_dh_trans())
    psi,theta,phi = puma.get_euler()
    x,y,z = puma.get_pose()
    print(f"\noutput:\n{x} {y} {z} {psi} {theta} {phi}")
    for i in range(3):
        print()

    matrix = np.array( [[ 0.10575416, -0.64251414, 0.75894113, 0.57764953],
                        [ 0.70190531,  0.58885882, 0.40071713, 0.36880972],
                        [-0.7043756,   0.49032731, 0.51325835, 0.19680029],
                        [ 0.,          0.,         0.,        1.,        ]])
    
    print(f"please enter Cartesian point:\ncol separate by ,\nrow separate by /\n")
    # 0.10575416,-0.64251414,0.75894113,0.57764953/0.70190531,0.58885882,0.40071713,0.36880972/-0.7043756,0.49032731,0.51325835,0.19680029/0.,0.,0.,1.
    if auto == "n":
        matrix_in = input()
        rows = re.split(r'/', matrix_in)
        matrix = []
        for r in rows:
            col = list(map(float, re.split(r',', r)))
            matrix.append(col)
    matrix = np.array(matrix)
    print(matrix)
    
    puma.dh_inverse_kinematics(matrix)


if __name__ == '__main__':
    main()