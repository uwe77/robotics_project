from visual_kinematics.RobotSerial import *
from visual_kinematics.Frame import *
import numpy as np
from math import pi
from robot_model import puma560
import re


def main():
    puma = puma560()
    # print("please enter the joint variable (in degree):")
    # for i in range(6):
    #     print(f"theta{i+1} {puma.joints[i]}", end=" ")
    # print(":")
    # rad = input()
    # rad = re.split(r'\s+', rad)
    # rad = [float(i) for i in rad]
    # print(rad)
    # puma.set_joint_angle(rad)
    # puma.get_dh_trans()
    # print("[n o a p]:\n", puma.dh_trans)
    # psi,theta,phi = puma.get_euler()
    # x,y,z = puma.get_pose()
    # print(f"\noutput:\n{x} {y} {z} {psi} {theta} {phi}")


    # matrix = []

    # for i in range(4):
    #     row = list(map(float, input("Enter row " + str(i+1) + " of the matrix, separated by spaces: ").split()))
    #     matrix.append(row)

    # Convert the list of lists to a numpy array
    matrix = np.array([[ 0.105, -0.643, 0.759, 0.583],
                        [ 0.702, 0.588, 0.4, 0.371],
                        [-0.704, 0.49, 0.513, 0.204],
                        [0., 0., 0., 1.]])

    print(matrix)
    end = Frame(matrix)
    puma.inverse(end)
    for i in range(6):
        print(f"theta{i+1}:{round(puma.dh_params[i,3]*180/pi, 6)}", end=" ")
    # print(puma.dh_params)
    print()
    puma.show()
    # puma.dh_inverse_kinematics(matrix)


if __name__ == '__main__':
    main()