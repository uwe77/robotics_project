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

def from_2_end(start, end):
    # Extract columns from A and B
    n1 = start[:, 0]
    o1 = start[:, 1]
    a1 = start[:, 2]
    p1 = start[:, 3]

    n2 = end[:, 0]
    o2 = end[:, 1]
    a2 = end[:, 2]
    p2 = end[:, 3]
    
    # Find solution for x, y, z
    x = np.dot(n1, (p2-p1))
    y = np.dot(o1, (p2-p1))
    z = np.dot(a1, (p2-p1))

    # Find solution for psi
    psi = atan2(dot(o1, a2), dot(o1, n2))
    
    # Find solution for theta
    theta = atan2(sqrt((n1@a2)**2 + (o1@a2)**2), a1@a2)

    # Find solution for phi
    s_psi = sin(psi)
    c_psi = cos(psi)
    v_theta = 1 - cos(theta)
    s_theta = sin(theta)
    c_theta = cos(theta)
    s_phi = -s_psi * c_psi * v_theta * (n1@n2) + ((c_psi)**2 * v_theta + c_theta) * (o1@n2) - s_psi * s_theta * (a1@n2)
    c_phi = -s_psi * c_psi * v_theta * (n1@o2) + ((c_psi)**2 * v_theta + c_theta) * (o1@o2) - s_psi * s_theta * (a1@o2)
    phi = atan2(s_phi, c_phi)
    state = {'x':x,
             'y':y,
             'z':z,
             'psi':psi,
             'theta':theta,
             'phi':phi}
    return state

def linear_trajectory(T, tsample, tacc, start, t1, t2, x, y, z, psi, theta, phi):
    # Define time vector
    t  = np.arange(0.0, 1.0, tsample)
    len_traj = t.shape[0]

    # Preallocate trajectory matrices
    trans_traj = np.zeros((4, 4, len_traj))
    pos_traj = np.zeros((3, len_traj))
    ori_traj = np.zeros((3, len_traj))

    for i in range(t.shape[0]):
        r = t[i] / T
        r_x = x * r
        r_y = y * r
        r_z = z * r
        r_theta = theta * r
        r_phi = phi * r

        s_psi = sin(psi)
        c_psi = cos(psi)
        v_r_theta = 1 - cos(r_theta)
        s_r_theta = sin(r_theta)
        c_r_theta = cos(r_theta)
        s_r_phi = sin(r_phi)
        c_r_phi = cos(r_phi)

        Tr = np.array([[ 1, 0, 0, r_x],
                        [0, 1, 0, r_y],
                        [0, 0, 1, r_z],
                        [0, 0, 0, 1  ]])

        Ra = np.array([[ s_psi**2 * v_r_theta + c_r_theta, -s_psi * c_psi * v_r_theta     , c_psi * s_r_theta, 0],
                        [-s_psi * c_psi * v_r_theta     , c_psi**2 * v_r_theta + c_r_theta, s_psi * s_r_theta, 0],
                        [-c_psi * s_r_theta             , -s_psi * s_r_theta             , c_r_theta        , 0],
                        [0                              , 0                              , 0                , 1]])

        Ro = np.array([[ c_r_phi, -s_r_phi,  0, 0],
                        [s_r_phi, c_r_phi ,  0, 0],
                        [0      , 0       ,  1, 0],
                        [0      , 0       ,  0, 1]])

        Dr = Tr@Ra@Ro
        trans_traj[:, :, i] = start@Dr
        pos_traj[:, i] = trans_traj[:3, 3, i]
        ori_traj[:, i] = trans_traj[:3, 2, i]
        state = {'trans_traj':trans_traj,
                 'pos_traj':pos_traj,
                 'ori_traj':ori_traj,
                 'len_traj':len_traj}
        return state

def trans_trajectory(T, tsample, tacc, start, t_start, t_end, xA, yA, zA, psiA, thetaA, phiA, xC, yC, zC, psiC, thetaC, phiC):
    # Define time vector
    t = np.arange(t_start, t_end, tsample)
    len_traj = t.shape[0]

    # Preallocate trajectory matrices
    trans_traj = np.zeros((4, 4, len_traj))
    pos_traj = np.zeros((3, len_traj))
    ori_traj = np.zeros((3, len_traj))

    for i in range(len_traj):
        h = (t[i] + tacc) / (2 * tacc)
        dx = ((xC * (tacc / T) + xA) * (2 - h) * h**2 - 2 * xA) * h + xA
        dy = ((yC * (tacc / T) + yA) * (2 - h) * h**2 - 2 * yA) * h + yA
        dz = ((zC * (tacc / T) + zA) * (2 - h) * h**2 - 2 * zA) * h + zA
        dpsi = (psiC - psiA) * h + psiA
        dtheta = ((thetaC * (tacc / T) + thetaA) * (2 - h) * h**2 - 2 * thetaA) * h + thetaA
        dphi = ((phiC * (tacc / T) + phiA) * (2 - h) * h**2 - 2 * phiA) * h + phiA

        s_psi = sin(dpsi)
        c_psi = cos(dpsi)
        v_theta = 1 - cos(dtheta)
        s_theta = sin(dtheta)
        c_theta = cos(dtheta)
        s_phi = sin(dphi)
        c_phi = cos(dphi)

        Tr = np.array([[ 1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1 ]])

        Ra = np.array([[ s_psi**2 * v_theta + c_theta  , -s_psi * c_psi * v_theta     , c_psi * s_theta, 0],
                        [-s_psi * c_psi * v_theta     , c_psi**2 * v_theta + c_theta  , s_psi * s_theta, 0],
                        [-c_psi * s_theta             , -s_psi * s_theta             , c_theta        , 0],
                        [ 0                           , 0                            , 0              , 1]])

        Ro = np.array([[ c_phi, -s_phi   ,  0, 0],
                        [s_phi, c_phi    ,  0, 0],
                        [0     , 0       ,  1, 0],
                        [0     , 0       ,  0, 1]])

        Dr = Tr@Ra @Ro
        trans_traj[:, :, i] = start * Dr
        pos_traj[:, i] = trans_traj[:3, 3, i]
        ori_traj[:, i] = trans_traj[:3, 2, i]
    
    state = {'trans_traj':trans_traj,
             'pos_traj':pos_traj,
             'ori_traj':ori_traj,
             'len_traj':len_traj}
    return state

def draw_cartesian(mode, t, c):
    if mode == 'value':
        title_plot = '   Position of xyz'
        yl = 'Position(cm)'
    elif mode == 'speed':
        title_plot = '   Velocity of xyz'
        yl = 'Velocity(cm/s)'
    elif mode == 'acc':
        title_plot = 'Accleration of xyz'
        yl = 'Acceleration(cm/s²)'
    else:
        title_plot = ''
        yl = ''
        print('Wrong type !!!')
    
    #plot joint angle
    fig, ax = plt.subplots(3,figsize=(10, 8))
    for sub in range(3):
        ax[sub].plot(range(c[sub].shape[0]), c[sub] , color='blue')
        ax[sub].set_title(f"{title_plot[:-4]} {title_plot[15+sub]}")
        ax[sub].set_xlabel('time (sec)')
        ax[sub].set_ylabel(yl)
    # plt.title(title_plot)
    plt.tight_layout()
    plt.show()

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
        if v <= (T-tacc): # A to A'
            j[:, i]  = P1 + (L1*v)
            js[:, i] = L1
            ja[:, i] = 0
        elif v >= (T + tacc): # A' to B
            j[:, i]  = P2 + (L2*(v-T))
            js[:, i] = L2
            ja[:, i] = 0
        else: # B' to C
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
    # linear A to A'
    A_B = from_2_end(A, B)
    A_ = linear_trajectory(T, tsample, tacc, A, 0, T-tacc, A_B['x'], A_B['y'], A_B['z'], A_B['psi'], A_B['theta'], A_B['phi'])
    # transition A' to B'
    A2 = A_['trans_traj'][:,:, A_['len_traj']-1]
    B_A2 = from_2_end(B, A2)

    B_C = from_2_end(B, C)

    if abs(B_C['psi']-B_A2['psi']) > pi/2:
        B_A2['psi'] += pi
        B_A2['theta'] *= -1
    
    A2_B2 = trans_trajectory(T, tsample, tacc, B, -tacc+tsample, tacc-tsample, B_A2['x'], B_A2['y'], B_A2['z'], B_A2['psi'], B_A2['theta'], B_A2['phi'], B_C['x'], B_C['y'], B_C['z'], B_C['psi'], B_C['theta'], B_C['phi'])
    
    # linear B' to C
    B2_C = linear_trajectory(T, tsample, tacc, B, tacc, T, B_C['x'], B_C['y'], B_C['z'], B_C['psi'], B_C['theta'], B_C['phi'])
    A_['pos_traj'] *= 100
    A2_B2['pos_traj'] *= 100
    B2_C['pos_traj'] *= 100

    # plot position
    t = np.arange(0.0, 1.0, tsample)

    # cx = np.array(A_['pos_traj'][0, :], A2_B2['pos_traj'][0, :], B2_C['pos_traj'][0, :])
    cx = np.hstack((A_['pos_traj'][0, :], A2_B2['pos_traj'][0, :], B2_C['pos_traj'][0, :]))
    cy = np.hstack((A_['pos_traj'][1, :], A2_B2['pos_traj'][1, :], B2_C['pos_traj'][1, :]))
    cz = np.hstack((A_['pos_traj'][2, :], A2_B2['pos_traj'][2, :], B2_C['pos_traj'][2, :]))
    c = np.array([cx, cy, cz])

    draw_cartesian('value', t, c)

    # Create figure for cartesian velocity
    dt = t[2:t.shape[0]]
    dcx = (cx[1:] - np.append(np.zeros((1)) ,cx[:-2])) / tsample
    dcy = (cy[1:] - np.append(np.zeros((1)) ,cy[:-2])) / tsample
    dcz = (cz[1:] - np.append(np.zeros((1)) ,cz[:-2])) / tsample
    dc = np.array([dcx, dcy, dcz])

    draw_cartesian('speed', dt, dc)


#     Create figure for cartesian accelerations
    dt2 = t[3:t.shape[0]]
    dc2x = (dcx[1:] - dcx[:-1]) / tsample
    dc2y = (dcy[1:] - dcy[:-1]) / tsample
    dc2z = (dcz[1:] - dcz[:-1]) / tsample
    dc2 = np.array([dc2x, dc2y, dc2z])

    draw_cartesian('acc', dt2, dc2)


#     % Create figure for cartesian path
#     title_plot = '3D path of cartesian space planning';
#     pos = c;
#     ori = [ori_traj_AA2(1, :), ori_traj_A2B2(1, :), ori_traj_B2C(1, :);
#          ori_traj_AA2(2, :), ori_traj_A2B2(2, :), ori_traj_B2C(2, :);
#          ori_traj_AA2(3, :), ori_traj_A2B2(3, :), ori_traj_B2C(3, :)];
    
#     draw_cartesian_path(title_plot, pos, ori);

# else
#     disp('Invalid mode. Please enter ''j'' or ''c''.');

def main():
    puma = puma560() # create a puma560 object
    # joint_move()
    cartesian_move()
    # cartesian_move()

if __name__ == '__main__':
    main()