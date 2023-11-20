from visual_kinematics.RobotSerial import*
from visual_kinematics.RobotTrajectory import *
from math import pi
from sympy import sin, cos
import sympy as sp
class Joint:
    def __init__(self, ary=[0,0,0,0], r = 0) -> None:
        self._params = ary
        self.range = r

    def __setitem__(self, index, value)->None:
        self._params[index] = value

    def __getitem__(self, index):
        return self._params[index]

    def __iter__(self):
        for i in self._params:
            yield i

    def __str__(self):
        return f"(-{self.range} ~ {self.range})"
       
    def set_theta(self, t)->None:
        self[3] = t

    def kinamatics(self):
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



class puma560(RobotSerial): # system ZYZ
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
    
    def __init__(self)->None:
        self.joint_params = np.array([[j for j in i] for i in self.joints])
        super().__init__(self.joint_params)
    
    def set_joint_angle(self, angles=[0,0,0,0,0,0])->None:
        for i in range(len(self.joints)):
            if self.joints[i].range < abs(angles[i]):
                print(f"theta{i+1} is out of range! default value: {self.joints[i][3]}")
            else:
                self.joints[i].set_theta(angles[i] * pi/180)
        self.dh_forward_kinematics()
        self.__init__()
        return (i.dh_matrix for i in self.joints)

    def dh_forward_kinematics(self):
        for i in self.joints:
            i.kinamatics()

    def get_dh_trans(self):
        self.dh_trans = np.eye(4)
        for i in self.joints:
            self.dh_trans = self.dh_trans @ i.dh_matrix
        return self.dh_trans
    
    def dh_inverse_kinematics(self, T):
        t1, t2, t3, t4, t5, t6 = sp.symbols('t1 t2 t3 t4 t5 t6')
        A1 = sp.Matrix([[cos(t1), 0., -sin(t1), 0.],
                        [sin(t1), 0.,  cos(t1), 0.],
                        [0.,     -1.,       0., 0.],
                        [0.,      0.,       0., 1.]])
        
        A2 = sp.Matrix([[cos(t2), -sin(t2), 0., 0.432*cos(t2)],
                        [sin(t2),  cos(t2), 0., 0.432*sin(t2)],
                        [0.,      0.,       1., 0.],
                        [0.,      0.,       0., 1.]])
        
        A3 = sp.Matrix([[cos(t3), 0.,  sin(t3), -0.02*cos(t3)],
                        [sin(t3), 0., -cos(t3), -0.02*sin(t3)],
                        [0.,      1.,       0.,  0.149],
                        [0.,      0.,       0.,  1.]])
        
        A4 = sp.Matrix([[cos(t4), 0., -sin(t4), 0.],
                        [sin(t4), 0.,  cos(t4), 0.],
                        [0.,     -1.,       0., 0.433],
                        [0.,      0.,       0., 1.]])

        A5 = sp.Matrix([[cos(t5), 0.,  sin(t5), 0.],
                        [sin(t5), 0., -cos(t5), 0.],
                        [0.,      1.,       0., 0.],
                        [0.,      0.,       0., 1.]])

        A6 = sp.Matrix([[cos(t6), -sin(t6), 0., 0.],
                        [sin(t6),  cos(t6), 0., 0.],
                        [0.,      0.,       1., 0.],
                        [0.,      0.,       0., 1.]])

        T6 = A1*A2*A3*A4*A5*A6
        solution = sp.solve((T6 - T), (t1, t2, t3, t4, t5, t6))
        print(solution)

    def get_euler(self):
        self.theta = np.arccos(self.dh_trans[2,2]) * 180/pi
        self.phi = np.arctan2(self.dh_trans[1,2], self.dh_trans[0,2]) * 180/pi
        self.psi = np.arctan2(self.dh_trans[2,1], -self.dh_trans[2,0]) * 180/pi
        return self.psi, self.theta, self.phi
    
    def get_pose(self):
        return self.dh_trans[0,3], self.dh_trans[1,3], self.dh_trans[2,3]
    
