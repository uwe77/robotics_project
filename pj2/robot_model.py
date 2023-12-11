import numpy as np
from math import pi


class Joint: # joint class
    def __init__(self, ary=[0,0,0,0], r = 0) -> None: # params: [d, a, alpha, theta] and range
        self._params = ary
        self.range = r

    def __setitem__(self, index, value)->None: # set params
        self._params[index] = value

    def __getitem__(self, index): # get params
        return self._params[index]

    def __iter__(self):
        for i in self._params:
            yield i

    def __str__(self): # fast print range
        return f"(-{self.range} ~ {self.range})"
       
    def set_theta(self, t)->None: # set theta
        self[3] = t

    def kinamatics(self): # calculate joint's transformation matrix
        c = lambda t: np.cos(t)
        s = lambda t: np.sin(t)
        t = self._params[3]
        a = self._params[2]
        al= self._params[1]
        d = self._params[0]
        matrix = [[c(t), -s(t)*c(a), s(t)*s(a), al*c(t)],
                [s(t), c(t)*c(a), -c(t)*s(a), al*s(t)],
                [0, s(a), c(a), d],
                [0, 0, 0, 1]]
        self.dh_matrix = np.array(matrix)
    
    def iv_dh_matrix(self): # calculate inverse transformation matrix
        return np.linalg.inv(self.dh_matrix)



class puma560(): # system ZYZ
    '''
    Joint   Link_offset   Link_Length   Twist_Angle theta
            mm            mm            rad         rad
    j1:     d1=0          a1=0          ap1=-pi/2   u1
    j2:     d2=0          a2=0.432      ap2= 0      u2
    j3:     d3=0.149      a3=-0.02      ap3= pi/2   u3
    j4:     d4=0.433      a4=0          ap4=-pi/2   u4
    j5:     d5=0          a5=0          ap5= pi/2   u5
    j6:     d6=0          a6=0          ap6=0       u6
    '''
    joints = [Joint([0., 0., -pi/2, 0.], 160),
            Joint([0., 0.432, 0., 0.], 125),
            Joint([0.149, -0.02 ,pi/2, 0.], 135),
            Joint([0.433, 0., -pi/2, 0.], 140),
            Joint([0., 0, pi/2, 0.], 100),
            Joint([0., 0., 0., 0.], 260)]
    
    def __init__(self)->None: # init
        for i in self.joints:
            i.kinamatics()
    
    def __setitem__(self, index, value)->None: # set each joint's theta
        value = np.arctan2(np.sin(value), np.cos(value)) # normalize angle
        if self.joints[index].range < abs(value)*180/pi: # check range
            print(f"theta{index+1} is out of range! range: {self.joints[index]}, value: {value*180/pi}")
        self.joints[index].set_theta(value) # set theta
        self.joints[index].kinamatics() # calculate dh matrix

    def __getitem__(self, index): # get each joint's theta
        return self.joints[index][3]

    def get_iv_trans_(self, index): # get inverse transform matrix
        T_final = np.eye(4)
        for i in range(index-1, -1, -1):# T^-1 = A(n)^-1 * A(n-1)^-1 ...
            T_final = T_final @ self.joints[i].iv_dh_matrix()
        return T_final

    def get_trans_(self, index1=1, index2=6): # get forward transform matrix from index1 to index2
        T_final = np.eye(4)
        for i in range(index1-1, index2-1, 1): # T = A(1) * A(2) * ... * A(n)
            T_final = T_final @ self.joints[i].dh_matrix
        return T_final

    def set_joints_angle(self, angles=[0,0,0,0,0,0])->None: # set all joints' theta
        for i in range(len(self.joints)):
            self[i] = angles[i]*pi/180

    def get_dh_trans(self): # get dh transform matrix(T6)
        self.dh_trans = self.get_trans_()
        return self.dh_trans
    
    def dh_inverse_kinematics(self, T): # inverse kinematics
        c = lambda t: np.cos(t)
        s = lambda t: np.sin(t)
        Px = T[0, 3]
        Py = T[1, 3]
        Pz = T[2, 3]
        times = 1
        for j in [1,-1]: # for theta3's two solutions
            for i in [1,-1]: # for theta1's two solutions
                for k in [0, 1]: # for theta4's two solutions
                    print(f"======times:{times}========")
                    # theta1's equation
                    self[0] = np.arctan2(Py, Px) - np.arctan2(self.joints[2][0], i*np.sqrt(Px**2 + Py**2 - self.joints[2][0]**2))
                    #                                                            +-
                    # theta3's equation
                    M = (Px**2 + Py**2 + Pz**2 - self.joints[1][1]**2 - self.joints[2][1]**2 - self.joints[2][0]**2 - self.joints[3][0]**2)/(2*self.joints[1][1])
                    self[2] = np.arctan2(M, j*np.sqrt(self.joints[2][1]**2 + self.joints[3][0]**2 - M**2)) - np.arctan2(self.joints[2][1], self.joints[3][0])
                    #                       +-
                    # theta2's equation
                    self[1] = np.arctan2(self.joints[3][0] + self.joints[1][1]*s(self[2]), self.joints[2][1]+self.joints[1][1]*c(self[2])) - np.arctan2(Pz, c(self[0])*Px + s(self[0])*Py) - self[2]
                    # theta4's equation
                    self[3] = np.arctan2(self.get_iv_trans_(3)[1]@T[:,2], self.get_iv_trans_(3)[0]@T[:,2]) + k*pi
                    # theta5's equation
                    self[4] = np.arctan2(self.get_iv_trans_(4)[0]@T[:,2], -self.get_iv_trans_(4)[1]@T[:,2])
                    # theta6's equation
                    self[5] = np.arctan2(self.get_iv_trans_(4)[2]@T[:,0], self.get_iv_trans_(4)[2]@T[:,1])
                    print("Corresponding Variable (theta1, theta2, theta3, theta4, theta5, theta6)")
                    for t in range(len(self.joints)):
                        print(self[t]*180/pi, end=" ")
                    print()
                    times += 1

    def get_euler(self): # get euler angle
        self.theta = np.arccos(self.dh_trans[2,2]) * 180/pi
        self.phi = np.arctan2(self.dh_trans[1,2], self.dh_trans[0,2]) * 180/pi
        self.psi = np.arctan2(self.dh_trans[2,1], -self.dh_trans[2,0]) * 180/pi
        return self.psi, self.theta, self.phi
    
    def get_pose(self): # get pose
        return self.dh_trans[0,3], self.dh_trans[1,3], self.dh_trans[2,3]