import numpy as np
from robot_model import puma560
import re
from numpy.linalg import inv, multi_dot
from numpy import dot,transpose
from math import pi, cos, sin, atan2, sqrt
import matplotlib.pyplot as plt

T = 0.5 # A-B and B-C time
tacc = 0.2 # for the transition portion
tsample = 0.002 # sample time
t  = np.arange(0.0, 1.0, tsample)
t1 = np.arange(-0.5, -0.2+0.002, tsample)
t2 = np.arange(-0.2+0.002, 0.2, tsample)
t3 = np.arange(0.2, 0.5, tsample)

A = np.array([[ 0.0064, 0.0077,   0.  ,  0.05],
              [ 0.0077,-0.0064,   0.  , -0.55],
              [   0.  ,   0.  ,  -0.01,  -0.6],
              [   0.  ,   0.  ,   0.  ,    1.]]) # m

B = np.array([[ 0.0087, -0.001, 0.0048,  0.5],
              [ 0.0029,  0.009,-0.0034, -0.4],
              [ -0.004, 0.0043, 0.0081,  0.4],
              [   0.  ,   0.  ,   0.  ,  1. ]]) # m

C = np.array([[ 0.0041,-0.0029, 0.0087,  0.6 ],
              [ 0.0069, 0.0071,-0.0009,  0.15],
              [ -0.006, 0.0064, 0.0049, -0.3 ],
              [   0.  ,   0.  ,   0.  ,   1. ]]) # m

def pos(x1, x2, x3, x4, x5, x6):
    puma = puma560() # create a puma560 object
    puma.set_joints_angle([x1, x2, x3, x4, x5, x6])
    # T1 = multi_dot( [A1(x1), A2(x2), A3(x3), A4(x4), A5(x5), A6(x6)])
    T1 = puma.get_dh_trans()
    x = T1[0,3]*100
    y = T1[1,3]*100
    z = T1[2,3]*100
    return x,y,z

def joint_move():
    P1 = np.array([-100.4577302516985, 70.61077411145472, 48.39965042684975, -5.957243485063329e-15, 60.98957546169554, 29.274572620257292])
    P2 = np.array([-52.11582808501224, -1.435829653814516, 30.20601749686518, 58.616557254148674, 11.478084623624586, 3.2858694580173395])
    P3 = np.array([0.09547312394060595, 65.79685457388466, 14.319573275123235, 15.33770637127752, -20.173045859599412, 29.516038512815136])

    L1 = (P2 - P1) / T
    L2 = (P3 - P2) / T

    P1c = P1 + (L1 * (T - tacc))

    dP2 = P1c - P2
    dP3 = P3 - P2

    j  = np.zeros((6, len(t)))
    js = np.zeros((6, len(t)))
    ja = np.zeros((6, len(t)))

    cp = np.zeros((3, len(t)))
    cv = np.zeros((3, len(t)))
    ca = np.zeros((3, len(t)))

    for i, v in enumerate(t):
        if v <= (T-tacc):
            j[:, i]  = P1 + (L1*v)
            js[:, i] = L1
            ja[:, i] = 0
        elif v >= (T + tacc):
            j[:, i]  = P2 + (L2*(v-T))
            js[:, i] = L2
            ja[:, i] = 0
        else:
            h = (v - T + tacc) / (2*tacc)
            j[:, i]  = ((dP3 * tacc / T + dP2)*(2-h)*(h**2) - 2*dP2)*h + P2 + dP2
            js[:, i] = ((dP3 * tacc / T + dP2)*(1.5-h)*2*(h**2) - dP2) / tacc
            ja[:, i] = (dP3 * tacc / T + dP2) *(1-h)*3*h/(tacc**2)            
        cp[0,i], cp[1,i], cp[2,i] = pos(j[0,i],j[1,i],j[2,i],j[3,i],j[4,i],j[5,i])
        
        if i != 0:
            cv[:,i] = (cp[:,i] - cp[:,i-1]) / tsample
        if i == 1:
            cv[:,0] = cv[:,1]
            ca[:,0] = ca[:,1]
        if i > 1:
            ca[:,i] = (cv[:,i] - cv[:,i-1])  / tsample

    #plot joint angle
    fig, ax = plt.subplots(3,2,figsize=(10, 8))
    for sub in range(6):
        ax[sub//2][sub%2].plot(t, j[sub,:] , color='blue')
        ax[sub//2][sub%2].set_title(f"Joint {sub+1}")
        ax[sub//2][sub%2].set_xlabel('time (sec)')
        ax[sub//2][sub%2].set_ylabel('angle (degree)')
    plt.title("Joint move - angle")
    plt.tight_layout()
    plt.show()

    #plot joint speed
    fig, ax = plt.subplots(3,2,figsize=(10, 8))
    for sub in range(6):
        ax[sub//2][sub%2].plot(t, js[sub,:] , color='blue')
        ax[sub//2][sub%2].set_title(f"Joint {sub+1}")
        ax[sub//2][sub%2].set_xlabel('time (sec)')
        ax[sub//2][sub%2].set_ylabel('angular velocity')
    plt.title("Joint move - angular velocity")
    plt.tight_layout()
    plt.show()

    #plot joint acc
    fig, ax = plt.subplots(3,2,figsize=(10, 8))
    for sub in range(6):
        ax[sub//2][sub%2].plot(t, ja[sub,:] , color='blue')
        ax[sub//2][sub%2].set_title(f"Joint {sub+1}")
        ax[sub//2][sub%2].set_xlabel('time (sec)')
        ax[sub//2][sub%2].set_ylabel('angular acceleration')
    plt.title("Joint move - angular acceleration")
    plt.tight_layout()
    plt.show()

    #3D
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(cp[0,:], cp[1,:], cp[2,:], label='3D path of Joint Move')
    ax.plot(A[0,3]*100,A[1,3]*100,A[2,3]*100, marker='o', markersize=5)
    ax.plot(B[0,3]*100,B[1,3]*100,B[2,3]*100, marker='o', markersize=5)
    ax.plot(C[0,3]*100,C[1,3]*100,C[2,3]*100, marker='o', markersize=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def cartesian_move():
    
    nA = np.array([A[0,0], A[1,0], A[2,0]])
    nA = nA.reshape(-1,1)
    oA = np.array([A[0,1], A[1,1], A[2,1]])
    oA = oA.reshape(-1,1)
    aA = np.array([A[0,2], A[1,2], A[2,2]])
    aA = aA.reshape(-1,1)
    pA = np.array([A[0,3], A[1,3], A[2,3]])
    pA = pA.reshape(-1,1)

    nB = np.array([B[0,0], B[1,0], B[2,0]])
    nB = nB.reshape(-1,1)
    oB = np.array([B[0,1], B[1,1], B[2,1]])
    oB = oB.reshape(-1,1)
    aB = np.array([B[0,2], B[1,2], B[2,2]])
    aB = aB.reshape(-1,1)
    pB = np.array([B[0,3], B[1,3], B[2,3]])
    pB = pB.reshape(-1,1)

    nC = np.array([C[0,0], C[1,0], C[2,0]])
    nC = nC.reshape(-1,1)
    oC = np.array([C[0,1], C[1,1], C[2,1]])
    oC = oC.reshape(-1,1)
    aC = np.array([C[0,2], C[1,2], C[2,2]])
    aC = aC.reshape(-1,1)
    pC = np.array([C[0,3], C[1,3], C[2,3]])
    pC = pC.reshape(-1,1)

    x = dot(nA.T, (pB - pA))
    y = dot(oA.T, (pB - pA))
    z = dot(aA.T, (pB - pA))
    psi = atan2(dot(oA.T, aB), dot(nA.T, aB))
    temp = sqrt(dot(nA.T, aB)**2 + dot(oA.T, aB)**2)
    theta = atan2(temp, dot(aA.T, aB))
    V_r_theta = 1-cos(theta)
    sin_phi = -sin(psi)*cos(psi)*V_r_theta*dot(nA.T, nB) + (cos(psi)**2*V_r_theta+cos(theta))*dot(oA.T, nB) - sin(psi)*sin(theta)*dot(aA.T, nB)
    cos_phi = -sin(psi)*cos(psi)*V_r_theta*dot(nA.T, oB) + (cos(psi)**2*V_r_theta+cos(theta))*dot(oA.T, oB) - sin(psi)*sin(theta)*dot(aA.T, oB)
    phi = atan2(sin_phi, cos_phi)

    # -0.5~-0.2 A~A'
    # pA_B = np.empty([4, 4])
    pA_B = []
    xA_B = []
    yA_B = []
    zA_B = []

    for i,v in enumerate(t1):
        h=(v+T)/T
        dx=float(x*h)
        dy=float(y*h)
        dz=float(z*h)
        dsi=psi
        dtheta=theta*h
        dphi=phi*h

        S_psi=sin(psi)
        C_psi=cos(psi)
        S_theta=sin(dtheta)
        C_theta=cos(dtheta)
        V_theta=1-C_theta
        S_phi=sin(dphi)
        C_phi=cos(dphi)
        Tr = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
        Rar = np.array([[S_psi**2*V_theta+C_phi, -S_psi*C_psi*V_theta,C_psi*S_phi, 0],
                        [-S_psi*C_psi*V_theta,C_psi**2*V_theta+C_phi, S_psi*S_phi, 0],
                        [-C_psi*S_phi, -S_psi*S_phi, C_phi, 0],
                        [0, 0, 0, 1]])
        Ror = np.array([[C_theta, -S_theta, 0, 0],
                        [S_theta, C_theta, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        Dr = multi_dot([Tr,Rar,Ror])
        # print(dot(a, Dr), " ", i)
        pA_B.append(dot(A, Dr))
        # print(pA_B, " ", i)
        xA_B.append(pA_B[i][0][3])
        yA_B.append(pA_B[i][1][3])
        zA_B.append(pA_B[i][2][3]) 

    #-0.2~0.2 A'~C'
    a2 = pA_B[-1]
    nA2 = np.array([a2[0,0], a2[1,0], a2[2,0]])
    nA2 = nA2.reshape(-1,1)
    oA2 = np.array([a2[0,1], a2[1,1], a2[2,1]])
    oA2 = oA2.reshape(-1,1)
    aA2 = np.array([a2[0,2], a2[1,2], a2[2,2]])
    aA2 = aA2.reshape(-1,1)
    pA2 = np.array([a2[0,3], a2[1,3], a2[2,3]])
    pA2 = pA2.reshape(-1,1)

    xA = dot(nB.T, (pA2-pB))
    yA = dot(oB.T, (pA2-pB))
    zA = dot(aB.T, (pA2-pB))
    psiA = atan2(dot(oB.T, aA2), dot(nB.T, aA2))
    thetaA = atan2(sqrt((dot(nB.T, aA2))**2 + dot(oB.T, aA2)**2), dot(aB.T, aA2))
    SphiA = -sin(psiA)*cos(psiA)*(1-cos(thetaA))*dot(nB.T, nA2)+((cos(psiA))**2*(1-cos(thetaA))+cos(thetaA))*dot(oB.T, nA2)-sin(psiA)*sin(thetaA)*dot(aB.T, nA2)
    CphiA = -sin(psiA)*cos(psiA)*(1-cos(thetaA))*dot(nB.T, oA2)+((cos(psiA))**2*(1-cos(thetaA))+cos(thetaA))*dot(oB.T, oA2)-sin(psiA)*sin(thetaA)*dot(aB.T, oA2)
    phiA = atan2(SphiA,CphiA)

    xC = dot(nB.T, (pC-pB))
    yC = dot(oB.T, (pC-pB))
    zC = dot(aB.T, (pC-pB))
    psiC = atan2(dot(oB.T, aC), dot(nB.T, aC))
    thetaC = atan2(sqrt(dot(nB.T, aC))**2 + dot(oB.T, aC)**2, dot(aB.T, aC))
    SphiC = -sin(psiC)*cos(psiC)*(1-cos(thetaC))*dot(nB.T, nC)+((cos(psiC))**2*(1-cos(thetaC))+cos(thetaC))*dot(oB.T, nC)-sin(psiC)*sin(thetaC)*dot(aB.T, nC)
    CphiC = -sin(psiC)*cos(psiC)*(1-cos(thetaC))*dot(nB.T, oC)+((cos(psiC))**2*(1-cos(thetaC))+cos(thetaC))*dot(oB.T, oC)-sin(psiC)*sin(thetaC)*dot(aB.T, oC)
    phiC=atan2(SphiC,CphiC)

    if abs(psiC-psiA)>pi/2:
        psiA = psiA+pi
        thetaA = -thetaA

    #path planing
    p_B = []
    x_B = []
    y_B = []
    z_B = []
    for i,v in enumerate(t2):
        h=(v+tacc)/(2*tacc)
        dx_B=float(((xC*tacc/T+xA)*(2-h)*h**2-2*xA)*h+xA)
        dy_B=float(((yC*tacc/T+yA)*(2-h)*h**2-2*yA)*h+yA)
        dz_B=float(((zC*tacc/T+zA)*(2-h)*h**2-2*zA)*h+zA)
        dpsi_B=float((psiC-psiA)*h+psiA)
        dtheta_B=float(((thetaC*tacc/T+thetaA)*(2-h)*h**2-2*thetaA)*h+thetaA)
        dphi_B=float(((phiC*tacc/T+phiA)*(2-h)*h**2-2*phiA)*h+phiA)

        S_psi=sin(dpsi_B)
        C_psi=cos(dpsi_B)
        S_theta=sin(dtheta_B)
        C_theta=cos(dtheta_B)
        V_theta=1-C_theta
        S_phi=sin(dphi_B)
        C_phi=cos(dphi_B)

        Tr = np.array([[1,0,0,dx_B],
                        [0,1,0,dy_B],
                        [0,0,1,dz_B],
                        [0,0,0,1]])
        Rar = np.array([[S_psi**2*V_theta+C_phi, -S_psi*C_psi*V_theta,C_psi*S_phi, 0],
                        [-S_psi*C_psi*V_theta,C_psi**2*V_theta+C_phi, S_psi*S_phi, 0],
                        [-C_psi*S_phi, -S_psi*S_phi, C_phi, 0],
                        [0, 0, 0, 1]])
        Ror = np.array([[C_theta, -S_theta, 0, 0],
                        [S_theta, C_theta, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        Dr_B = multi_dot([Tr,Rar,Ror])

        p_B.append(dot(B, Dr_B))
        x_B.append(p_B[i][0][3])
        y_B.append(p_B[i][1][3])
        z_B.append(p_B[i][2][3]) 

    # 0.2~0.5 C'~C
    p_C = []
    x_C = []
    y_C = []
    z_C = []
    for i,v in enumerate(t3):
        h=v/0.5
        dx_C=float(xC*h)
        dy_C=float(yC*h)
        dz_C=float(zC*h)
        dpsi_C=psiC
        dtheta_C=thetaC*h
        dphi_C=phiC*h

        S_psi=sin(dpsi_C)
        C_psi=cos(dpsi_C)
        S_theta=sin(dtheta_C)
        C_theta=cos(dtheta_C)
        V_theta=1-C_theta
        S_phi=sin(dphi_C)
        C_phi=cos(dphi_C)

        Tr = np.array([[1,0,0,dx_C],
                        [0,1,0,dy_C],
                        [0,0,1,dz_C],
                        [0,0,0,1]])
        Rar = np.array([[S_psi**2*V_theta+C_phi, -S_psi*C_psi*V_theta,C_psi*S_phi, 0],
                        [-S_psi*C_psi*V_theta,C_psi**2*V_theta+C_phi, S_psi*S_phi, 0],
                        [-C_psi*S_phi, -S_psi*S_phi, C_phi, 0],
                        [0, 0, 0, 1]])
        Ror = np.array([[C_theta, -S_theta, 0, 0],
                        [S_theta, C_theta, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        Dr_C = multi_dot([Tr,Rar,Ror])

        p_C.append(dot(B, Dr_C))
        x_C.append(p_C[i][0][3])
        y_C.append(p_C[i][1][3])
        z_C.append(p_C[i][2][3]) 

    #3d
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(np.concatenate((xA_B, x_B, x_C), axis=None), np.concatenate((yA_B, y_B, y_C), axis=None), np.concatenate((zA_B, z_B, z_C), axis=None), label='3D path of Cartesian Move')
    ax.plot(A[0,3],A[1,3],A[2,3], marker='o', markersize=5)
    ax.plot(B[0,3],B[1,3],B[2,3], marker='o', markersize=5)
    ax.plot(C[0,3],C[1,3],C[2,3], marker='o', markersize=15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    #xyz position
    Xconcate = np.concatenate((xA_B, x_B, x_C), axis=None)
    Yconcate = np.concatenate((yA_B, y_B, y_C), axis=None)
    Zconcate = np.concatenate((zA_B, z_B, z_C), axis=None)
    _position = [Xconcate, Yconcate, Zconcate]
    fig, ax = plt.subplots(3,1,figsize=(10, 8))
    for sub in range(3):
        ax[sub].plot(t[:], _position[sub] , color='blue')
        ax[sub].set_title(f"Position {sub+1}")
        ax[sub].set_xlabel('time (sec)')
        ax[sub].set_ylabel('position(cm)')
    plt.title("Position")
    plt.tight_layout()
    plt.show()

    #xyz velocity
    dX = np.diff(Xconcate) / tsample
    dY = np.diff(Yconcate) / tsample
    dZ = np.diff(Xconcate) / tsample
    _dv = [dX, dY, dZ]
    fig, ax = plt.subplots(3,1,figsize=(10, 8))
    for sub in range(3):
        ax[sub].plot(t[1:], _dv[sub] , color='blue')
        ax[sub].set_title(f"Velocity {sub+1}")
        ax[sub].set_xlabel('time (sec)')
        ax[sub].set_ylabel('velocity(cm/s)')
    plt.title("Velocity")
    plt.tight_layout()
    plt.show()

    #xyz acceleration
    ddX = np.diff(dX) / tsample
    ddY = np.diff(dY) / tsample
    ddZ = np.diff(dZ) / tsample
    _da = [ddX, ddY, ddZ]
    fig, ax = plt.subplots(3,1,figsize=(10, 8))
    for sub in range(3):
        ax[sub].plot(t[2:], _da[sub] , color='blue')
        ax[sub].set_title(f"acceleration {sub+1}")
        ax[sub].set_xlabel('time (sec)')
        ax[sub].set_ylabel('acceleration(cm/s^2)')
    plt.title("acceleration")
    plt.tight_layout()
    plt.show()

def main():
    puma = puma560() # create a puma560 object
    # joint_move()
    cartesian_move()

if __name__ == '__main__':
    main()