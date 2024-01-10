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


A = np.array([[ 0.64, 0.77, 0.  , 0.05],
              [ 0.77,-0.64, 0.  ,-0.55],
              [ 0.  , 0.  ,-1   ,-0.6 ],
              [ 0.  , 0.  , 0.  , 1.  ]]) # m

B = np.array([[ 0.87,-0.1 , 0.48, 0.5],
              [ 0.29, 0.9 ,-0.34,-0.4],
              [-0.4 , 0.43, 0.81, 0.4],
              [ 0.  , 0.  , 0.  , 1. ]]) # m

C = np.array([[ 0.41,-0.29, 0.87,  0.6 ],
              [ 0.69, 0.71,-0.09,  0.15],
              [-0.6 , 0.64, 0.49, -0.3 ],
              [ 0.  , 0.  , 0.  ,  1.  ]]) # m

def trans_pos(x1, x2, x3, x4, x5, x6):
    puma = puma560() # create a puma560 object
    puma.set_joints_angle([x1, x2, x3, x4, x5, x6])
    # T1 = multi_dot( [A1(x1), A2(x2), A3(x3), A4(x4), A5(x5), A6(x6)])
    T1 = puma.get_dh_trans()
    x = T1[0,3]*100
    y = T1[1,3]*100
    z = T1[2,3]*100
    psi, theta, phi = puma.get_euler()
    return x,y,z, psi,theta,phi

def from_2_end(start, end):
    # Extract columns from A and B
    n1 = start[:3, 0]
    o1 = start[:3, 1]
    a1 = start[:3, 2]
    p1 = start[:3, 3]

    n2 = end[:3, 0]
    o2 = end[:3, 1]
    a2 = end[:3, 2]
    p2 = end[:3, 3]
    
    # Find solution for x, y, z
    x = n1@(p2-p1)
    y = o1@(p2-p1)
    z = a1@(p2-p1)

    # Find solution for psi
    psi = atan2(o1@a2, n1@a2) # atan2(o1.T@a2, o1@n2)
    
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

def linear_trajectory(T, tsample, tacc, start, t1, t2, trans:dict):
    # Define time vector
    t = np.arange(t1, t2+tsample, tsample)
    len_traj = t.shape[0]

    # Preallocate trajectory matrices
    trans_traj = np.zeros((len_traj, 4, 4))
    pos_traj = np.zeros((len_traj, 3))
    ori_traj = np.zeros((len_traj, 3))

    for i, v in enumerate(t):
        r = v / T
        r_x = trans['x'] * r
        r_y = trans['y'] * r
        r_z = trans['z'] * r
        r_theta = trans['theta'] * r
        r_phi = trans['phi'] * r

        s_psi = sin(trans['psi'])
        c_psi = cos(trans['psi'])
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
                        [-c_psi * s_r_theta             , -s_psi * s_r_theta              , c_r_theta        , 0],
                        [0                              , 0                               , 0                , 1]])

        Ro = np.array([[ c_r_phi, -s_r_phi,  0, 0],
                        [s_r_phi, c_r_phi ,  0, 0],
                        [0      , 0       ,  1, 0],
                        [0      , 0       ,  0, 1]])

        Dr = Tr@Ra@Ro
        trans_traj[i, :, :] = start@Dr
        pos_traj[i, :] = trans_traj[i, :3, 3]
        ori_traj[i, :] = trans_traj[i, :3, 2]
        state = {'trans_traj':trans_traj,
                 'pos_traj':pos_traj,
                 'ori_traj':ori_traj,
                 'len_traj':len_traj}
    return state

def trans_trajectory(T, tsample, tacc, start, t1, t2, A:dict, C:dict):
    # Define time vector
    t = np.arange(t1, t2+tsample, tsample)
    len_traj = t.shape[0]
    # Preallocate trajectory matrices
    trans_traj = np.zeros((len_traj, 4, 4))
    pos_traj = np.zeros((len_traj, 3))
    ori_traj = np.zeros((len_traj, 3))

    for i, v in enumerate(t):
        h = (v + tacc) / (2 * tacc)
        dx = ((C['x'] * (tacc / T) + A['x']) * (2 - h) * h**2 - 2 * A['x']) * h + A['x']
        dy = ((C['y'] * (tacc / T) + A['y']) * (2 - h) * h**2 - 2 * A['y']) * h + A['y']
        dz = ((C['z'] * (tacc / T) + A['z']) * (2 - h) * h**2 - 2 * A['z']) * h + A['z']
        dpsi = (C['psi'] - A['psi']) * h + A['psi']
        dtheta = ((C['theta'] * (tacc / T) + A['theta']) * (2 - h) * h**2 - 2 * A['theta']) * h + A['theta']
        dphi = ((C['phi'] * (tacc / T) + A['phi']) * (2 - h) * h**2 - 2 * A['phi']) * h + A['phi']

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

        # Dr = Tr@Ra @Ro
        trans_traj[i, :, :] = start@Tr@Ra@Ro
        pos_traj[i, :] = trans_traj[i, :3, 3]
        ori_traj[i, :] = trans_traj[i, :3, 2]
    
    state = {'trans_traj':trans_traj,
             'pos_traj':pos_traj,
             'ori_traj':ori_traj,
             'len_traj':len_traj}
    return state

def draw_cartesian(mode, t, c):
    xyz='xyz'
    if mode == 'value':
        title_plot = 'Position----of'
        yl = 'Position(cm)'
    elif mode == 'speed':
        title_plot = 'Velocity----of'
        yl = 'Velocity(cm/s)'
    elif mode == 'acc':
        title_plot = 'Accleration-of'
        yl = 'Acceleration(cm/s²)'
    else:
        title_plot = ''
        yl = ''
        print('Wrong type !!!')
    
    #plot joint angle
    fig, ax = plt.subplots(3,figsize=(10, 10))
    for sub in range(3):
        ax[sub].plot(t, c[sub] , color='blue')
        ax[sub].set_title(f"{title_plot}-{xyz[sub]}")
        ax[sub].set_xlabel('time (sec)')
        ax[sub].set_ylabel(yl)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"./images/Cartesian-move-{title_plot}.png")
    input("Press Enter to close the plot...")
    plt.close()

def draw_joint(mode, t, c):
    if mode == 'angle':
        title_plot = 'angle-degree'
        yl = 'Position(cm)'
    elif mode == 'speed':
        title_plot = 'angular-speed'
        yl = 'Velocity(cm/s)'
    elif mode == 'acc':
        title_plot = 'angular-acceleration'
        yl = 'Acceleration(cm/s²)'
    else:
        title_plot = ''
        yl = ''
        print('Wrong type !!!')
    
    #plot joint angle
    fig, ax = plt.subplots(3,2,figsize=(10, 8))
    for sub in range(6):
        ax[sub//2][sub%2].plot(t, c[sub,:] , color='blue')
        ax[sub//2][sub%2].set_title(f"Joint {sub+1}")
        ax[sub//2][sub%2].set_xlabel('time (sec)')
        ax[sub//2][sub%2].set_ylabel(yl)
    # plt.title(f"Joint move - {title_plot}")
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig(f"./images/Joint-move-{title_plot}.png")
    input("Press Enter to close the plot...")
    plt.close()
    
def draw_cartesian_path(title_plot, pos, ori, mode):
    length = 0.05
    if mode == 'c':
        length = 5
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(A[0,3]*100,A[1,3]*100,A[2,3]*100, marker='o', markersize=5)
    ax.plot(B[0,3]*100,B[1,3]*100,B[2,3]*100, marker='o', markersize=10)
    ax.plot(C[0,3]*100,C[1,3]*100,C[2,3]*100, marker='o', markersize=15)
    ax.plot(pos[0, :], pos[1, :], pos[2, :], label='3D path of cartesian space planning', color='red', linewidth=4)
    ax.quiver(pos[0, :], pos[1, :], pos[2, :], ori[0, :], ori[1, :], ori[2, :], length=length, label='orientation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title_plot)
    plt.show(block=False)
    input("Press Enter to close the plot...")
    plt.close()


def joint_move():
    t  = np.arange(0.0, 2*T, tsample)
    # point A got only one solution 
    P1 = np.array([-100.4577302516985, 70.61077411145472, 48.39965042684975, -5.957243485063329e-15, 60.98957546169554, 29.274572620257292])
    
    # point B got three solutions
    # P2 = np.array([-52.11582808501224, -1.435829653814516, 30.20601749686518, 58.616557254148674, 11.478084623624586, 3.2858694580173395])
    P2 = np.array([-52.11582808501224, -1.435829653814516, 30.20601749686518, -121.38344274585134, -11.478084623624586, -176.71413054198266])
    # P2 = np.array([154.79621157683206, -116.00831644153826, 30.20601749686518, 7.680699442471632, 50.49405898570072, -145.53930119744948])
    
    # point C got only one solution
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
    ori = np.zeros((3, len(t)))

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
        cp[0,i], cp[1,i], cp[2,i], ori[0,i], ori[1,i], ori[2,i] = trans_pos(j[0,i],j[1,i],j[2,i],j[3,i],j[4,i],j[5,i])
        
        if i != 0:
            cv[:,i] = (cp[:,i] - cp[:,i-1]) / tsample
        if i == 1:
            cv[:,0] = cv[:,1]
            ca[:,0] = ca[:,1]
        if i > 1:
            ca[:,i] = (cv[:,i] - cv[:,i-1])  / tsample

    #plot joint angle
    draw_joint('angle', t, j)

    #plot joint speed
    draw_joint('speed', t, js)

    #plot joint acc
    draw_joint('acc', t, ja)

    #3D
    draw_cartesian_path('3D path of joint space planning', cp, ori, 'j')

def cartesian_move():

    AB = from_2_end(A, B)
    BC = from_2_end(B, C)
    # -0.5~-0.2 A~A'
    A_ = linear_trajectory(T, tsample, tacc, A, 0., T-tacc, AB)
    # -0.2~0.2 A'~B'
    Ae = A_['trans_traj'][-1, :, :]
    B_Ae = from_2_end(B, Ae)
    if abs(BC['psi']-B_Ae['psi']) > pi/2:
        B_Ae['psi'] += pi
        B_Ae['theta'] *= -1
    B_ = trans_trajectory(T, tsample, tacc, B, -tacc+tsample, tacc-tsample, B_Ae, BC)
    # 0.2~0.5 B'~C
    C_ = linear_trajectory(T, tsample, tacc, B, tacc, T, BC)

    A_['pos_traj'] *= 100
    B_['pos_traj'] *= 100
    C_['pos_traj'] *= 100
    # plot position
    t = np.arange(0., 2.0*T+tsample, tsample)

    # cx = np.array(A_['pos_traj'][0, :], A2_B2['pos_traj'][0, :], B2_C['pos_traj'][0, :])
    cx = np.hstack((A_['pos_traj'][:, 0], B_['pos_traj'][:, 0], C_['pos_traj'][:, 0]))
    cy = np.hstack((A_['pos_traj'][:, 1], B_['pos_traj'][:, 1], C_['pos_traj'][:, 1]))
    cz = np.hstack((A_['pos_traj'][:, 2], B_['pos_traj'][:, 2], C_['pos_traj'][:, 2]))
    c = np.array([cx, cy, cz])
    draw_cartesian('value', t, c)

    # Create figure for cartesian velocity
    dt = t[:-2]
    dcx = (np.append(cx[1:], cx[-1])[:-1] - cx[:-1]) / tsample
    dcy = (np.append(cy[1:], cy[-1])[:-1] - cy[:-1]) / tsample
    dcz = (np.append(cz[1:] ,cz[-1])[:-1] - cz[:-1]) / tsample
    dcx = np.concatenate((dcx[:150], dcx[150+1:]))
    dcy = np.concatenate((dcy[:150], dcy[150+1:]))
    dcz = np.concatenate((dcz[:150], dcz[150+1:]))
    dc = np.array([dcx, dcy, dcz])
    draw_cartesian('speed', dt, dc)

    # Create figure for cartesian accelerations
    ddt = t[:-4]
    ddcx = (np.append(dcx[1:], dcx[-1])[:-1] - dcx[:-1]) / tsample
    ddcy = (np.append(dcy[1:], dcy[-1])[:-1] - dcy[:-1]) / tsample
    ddcz = (np.append(dcz[1:], dcz[-1])[:-1] - dcz[:-1]) / tsample
    ddcx = np.concatenate((ddcx[:149], ddcx[150:]))
    ddcy = np.concatenate((ddcy[:149], ddcy[150:]))
    ddcz = np.concatenate((ddcz[:149], ddcz[150:]))
    dc2 = np.array([ddcx, ddcy, ddcz])
    draw_cartesian('acc', ddt, dc2)

    # Create figure for cartesian path
    title_plot = '3D path of cartesian space planning'
    pos = c
    ori = np.array([np.hstack((A_['ori_traj'][:, 0], B_['ori_traj'][:, 0], C_['ori_traj'][:, 0])),
                    np.hstack((A_['ori_traj'][:, 1], B_['ori_traj'][:, 1], C_['ori_traj'][:, 1])),
                    np.hstack((A_['ori_traj'][:, 2], B_['ori_traj'][:, 2], C_['ori_traj'][:, 2]))])
    draw_cartesian_path(title_plot, pos, ori, 'c')

def main():
    # puma = puma560()
    # puma.dh_inverse_kinematics(A)
    # puma.dh_inverse_kinematics(B)
    # puma.dh_inverse_kinematics(C)
    while KeyboardInterrupt:
        mode = input("Enter mode (1: joint move, 2: cartesian move, q: leave): ")
        if mode == '1':
            joint_move()
        elif mode == '2':
            cartesian_move()
        elif mode == 'q':
            break

if __name__ == '__main__':
    main()