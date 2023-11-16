from visual_kinematics.RobotSerial import*
from visual_kinematics.RobotTrajectory import *
from math import pi

class Joint:
    def __init__(self, ary=[0,0,0,0]) -> None:
        self._params = ary
    
    def __setitem__(self, index, value)->None:
        self._params[index] = value

    def __getitem__(self, index):
        return self._params[index]

    def __iter__(self):
        for i in self._params:
            yield i

    def set_angle(self, a)->None:
        self[0]

class puma560:
    '''
    Joint   Link_offset   Link_Length   Twist_Angle theta
            mm            mm            rad         rad
    j1:     d1=0          a1=0          ap1= pi/2   u1
    j2:     d2=0          a2=43.18      ap2= 0      u2
    j3:     d3=15         a3=2.03       ap3=-pi/2   u3
    j4:     d4=43.18      a4=0          ap4= pi/2   u4
    j5:     d5=0          a5=0          ap5=-pi/2   u5
    j6:     d6=0          a6=0          ap6=0       u6
    '''
    j1 = Joint([0., 0., pi/2, 0.])
    j2 = Joint([0., 0.4318, 0., 0.])
    j3 = Joint([0.15, 0.0203 ,-pi/2, 0.])
    j4 = Joint([0.4318, 0., pi/2, 0.])
    j5 = Joint([0., 0, -pi/2, 0.])
    j6 = Joint([0., 0., 0., 0.])
    
    def __init__(self)->None:
        self.dh_params = np.array([[i for i in self.j1],
                            [i for i in self.j2],
                            [i for i in self.j3],
                            [i for i in self.j4],
                            [i for i in self.j5],
                            [i for i in self.j6]])
        self.robot = RobotSerial(self.dh_params)
        
    def set_roate(self)->None:
        frames = [Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [0.63182]])),
              Frame.from_euler_3(np.array([0.25 * pi, 0., 0.75 * pi]), np.array([[0.48127], [0.], [0.63182]])),
              Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.63182]])),
              Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.48127], [0.], [0.23182]]))]
        trajectory = RobotTrajectory(self.robot, frames)
        trajectory.show(motion="p2p")