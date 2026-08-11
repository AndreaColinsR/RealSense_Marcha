import numpy as np
import matplotlib.pyplot as plt
from utils import DLT, kalman_acceleration_smooth, postprocess_3d_keypoints,define_steps_middle,define_steps
import cv2 as cv
from scipy.signal import find_peaks
#from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-v0_8')


pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 32, 31])
Npoints = len(pose_keypoints)

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Finds angle between two vectors"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_signed(v1, v2):
    """Finds signed angle between two 2D vectors (in degrees, or radians if you drop degrees())"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    cross = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]   # z-component of cross product
    dot = np.dot(v1_u, v2_u)
    return np.arctan2(cross, dot)

def fix_sign_for_walking_direction(angle_array, position_marker):
    """
    Flips the sign of a signed angle array if the person is walking
    in the negative x-direction (against the x-axis).

    Parameters
    ----------
    angle_array : array-like
        The signed angle values for a step (e.g., output of angle_between_signed).
    position_marker : array-like
        x-coordinates of a marker over the same step (e.g., pelvis, sacrum,
        or heel marker) — used to determine walking direction.

    Returns
    -------
    np.ndarray
        Angle array with sign corrected so that all steps share the same
        walking-direction convention.
    """
    angle_array = np.asarray(angle_array)
    position_marker = np.asarray(position_marker)

    # net displacement in x over the step: positive = walking in +x, negative = walking in -x
    net_displacement = position_marker[-1] - position_marker[0]

    if net_displacement < 0:
        # walking against the x-axis -> flip sign to match the +x convention
        return -angle_array
    else:
        return angle_array

def get_steps(steps_start,angles):
    angle_steps=[]
    for i in range(len(steps_start)):
        this_step = angles[steps_start[i][0]:steps_start[i][1]+1]*180/np.pi
        angle_steps.append(this_step)
    
    return angle_steps
    
def visualize_3d(p3ds,capF,capS,t):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    anklel = [[10, 12]]
    ankler = [[11, 13]]
    body = [torso, arml, armr, legr, legl , anklel,ankler]
    colors = ['red', 'blue', 'green', 'black', 'orange','cyan','magenta']


    counter=1
    tmp=p3ds.shape
    Nframes=tmp[0]
    t=t[0:Nframes]


    tmp_nan=[];
    show_frame=[];
    
    for i in range(Nframes):
        if np.sum(p3ds[i,0,:])>-3.1 and np.sum(p3ds[i,0,:])<-2.9:
            tmp_nan.append(i)
        else:
            show_frame.append(i)
            
    p3ds=np.delete(p3ds,tmp_nan, axis=0)
    t = np.delete(t, tmp_nan)
    
    p3ds = p3ds-p3ds[0,10,:]
    
    tmp = p3ds.shape
    Nframes=tmp[0]
    l_knee=np.zeros((Nframes,))
    r_knee=np.zeros((Nframes,))

    l_hip=np.zeros((Nframes,))
    r_hip=np.zeros((Nframes,))

    l_arm=np.zeros((Nframes,))
    r_arm=np.zeros((Nframes,))
    
    l_ankle=np.zeros((Nframes,))
    r_ankle=np.zeros((Nframes,))

    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])

    for x_1 in range(0,3):
        for kpoint in range(0,Npoints):
            p3ds[:,kpoint,x_1]=kalman_acceleration_smooth(p3ds[:,kpoint,x_1], process_noise=1e-2, measurement_noise=1e-4)
    
    for i in range(Nframes):
        # ankle 
        #x_aligned=np.array([p3ds[i,12,0],p3ds[i,12,1],p3ds[i,10,2]])
        l_ankle[i]=angle_between(p3ds[i,12,:]-p3ds[i,10,:], p3ds[i,8,:]-p3ds[i,10,:])-np.pi/2
        #l_ankle[i]=angle_between(p3ds[i,12,[0,2]]-p3ds[i,10,[0,2]], p3ds[i,8,[0,2]]-p3ds[i,10,[0,2]])-np.pi/2
        r_ankle[i]=angle_between(p3ds[i,13,:]-p3ds[i,11,:], p3ds[i,9,:]-p3ds[i,11,:])-np.pi/2
        
        # knee    
        l_knee[i]=angle_between(p3ds[i,6,:]-p3ds[i,8,:], p3ds[i,8,:]-p3ds[i,10,:])
        r_knee[i]=angle_between(p3ds[i,7,:]-p3ds[i,9,:], p3ds[i,9,:]-p3ds[i,11,:])
        # hip
        #print(p3ds[i,0,:]-p3ds[i,6,:])
        #print([0,0,-1])
        #plt.pause(1)
        zaligned=np.array([p3ds[i,6,0],p3ds[i,6,1],0])
        l_hip[i]=angle_between_signed(zaligned[[0,2]]-p3ds[i,6,[0,2]], p3ds[i,8,[0,2]]-p3ds[i,6,[0,2]])
        zaligned=np.array([p3ds[i,7,0],p3ds[i,7,1],p3ds[i,9,2]])
        r_hip[i]=angle_between_signed(zaligned[[0,2]]-p3ds[i,7,[0,2]], p3ds[i,9,[0,2]]-p3ds[i,7,[0,2]])
        
        #shoulder
        l_arm[i]=angle_between(z, p3ds[i,1,:]-p3ds[i,3,:])
        r_arm[i]=angle_between(z, p3ds[i,0,:]-p3ds[i,2,:])


    ## FIgure of angles for each cycle of gait
    fig0 = plt.figure()

    ###### Detect steps for right leg
   
    start_steps,t_steps=define_steps_middle(r_knee,t)
    steps_r_knee=get_steps(start_steps,r_knee)
    steps_r_hip=get_steps(start_steps,r_hip)
    steps_r_ankle=get_steps(start_steps,r_ankle)
    steps_pelvis_x=get_steps(start_steps,p3ds[:,6,0])

    Nsteps=len(steps_r_knee)

    n_points = 101  # common grid: 0% to 100% gait cycle
    x_common = np.linspace(0, 100, n_points)

    r_knee_interp = np.zeros((Nsteps, n_points))
    r_hip_interp = np.zeros((Nsteps, n_points))
    r_ankle_interp = np.zeros((Nsteps, n_points))
    
    for i in range(Nsteps):
       
        this_color=i/(Nsteps+3)
        x_percent = np.linspace(0, 100, len(steps_r_knee[i]))
        
        plt.subplot(231)
        plt.plot(x_percent, steps_r_knee[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo rodilla derecha [o]')

        plt.subplot(232)
        steps_r_hip[i] = fix_sign_for_walking_direction(steps_r_hip[i], steps_pelvis_x[i])
        plt.plot(x_percent, steps_r_hip[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo cadera derecha [o]')

        plt.subplot(233)
        plt.plot(x_percent, steps_r_ankle[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo tobillo derecha [o]')

        # interpolate this step onto the common grid for later averaging
        r_knee_interp[i, :] = np.interp(x_common, x_percent, steps_r_knee[i])
        r_hip_interp[i, :] = np.interp(x_common, x_percent, steps_r_hip[i])
        r_ankle_interp[i, :] = np.interp(x_common, x_percent, steps_r_ankle[i])

        # compute averages
    r_knee_mean = r_knee_interp.mean(axis=0)
    r_hip_mean = r_hip_interp.mean(axis=0)
    r_ankle_mean = r_ankle_interp.mean(axis=0)

    # plot averages on top, in a distinct color
    plt.subplot(231)
    plt.plot(x_common, r_knee_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-10, 90)

    plt.subplot(232)
    plt.plot(x_common, r_hip_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-50, 50)

    plt.subplot(233)
    plt.plot(x_common, r_ankle_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-50, 50)
    
    ## compute for left leg
    start_steps,t_steps=define_steps_middle(l_knee,t)
    steps_l_knee=get_steps(start_steps,l_knee)
    steps_l_hip=get_steps(start_steps,l_hip)
    steps_l_ankle=get_steps(start_steps,l_ankle)

    Nsteps=len(steps_l_knee)

    l_knee_interp = np.zeros((Nsteps, n_points))
    l_hip_interp = np.zeros((Nsteps, n_points))
    l_ankle_interp = np.zeros((Nsteps, n_points))

    for i in range(Nsteps):
        this_color=i/(Nsteps+3)
        x_percent = np.linspace(0, 100, len(steps_l_knee[i]))

        plt.subplot(234)
        plt.plot(x_percent, steps_l_knee[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo rodilla izquierda [o]')

        plt.subplot(235)
        steps_l_hip[i] = fix_sign_for_walking_direction(steps_l_hip[i], steps_pelvis_x[i])
        plt.plot(x_percent, steps_l_hip[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo cadera izquierda [o]')

        plt.subplot(236)
        plt.plot(x_percent, steps_l_ankle[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo tobillo izquierda [o]')

        # interpolate this step onto the common grid for later averaging
        l_knee_interp[i, :] = np.interp(x_common, x_percent, steps_l_knee[i])
        l_hip_interp[i, :] = np.interp(x_common, x_percent, steps_l_hip[i])
        l_ankle_interp[i, :] = np.interp(x_common, x_percent, steps_l_ankle[i])
        
    # compute averages
    l_knee_mean = l_knee_interp.mean(axis=0)
    l_hip_mean = l_hip_interp.mean(axis=0)
    l_ankle_mean = l_ankle_interp.mean(axis=0)

    # plot averages on top, in a distinct color
    plt.subplot(234)
    plt.plot(x_common, l_knee_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-10, 90)
    
    plt.subplot(235)
    plt.plot(x_common, l_hip_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-50, 50)
    
    plt.subplot(236)
    plt.plot(x_common, l_ankle_mean, color='red', linewidth=2, label='Promedio')
    plt.ylim(-50, 50)

    fig1 = plt.figure()
    for i in range(Nsteps):
        
        plt.subplot(233)
        plt.plot(1,t_steps[i],marker='o',color=[0.5,0.5,0.5])
        
    
    t_steps2=np.array(t_steps)
    print("Promedio duracion de paso: ", np.mean(t_steps2))
    plt.errorbar(1,np.mean(t_steps2),np.std(t_steps2),fmt='o-',color='k')
    plt.ylabel('Tiempo de cada paso [s]') 

    ## plot angles over time 
    fig0 = plt.figure()
    plt.subplot(411)
    plt.plot(t[1:len(t)],r_hip[1:len(t)]*180/np.pi,'r',label='Cadera derecha')
    plt.plot(t[1:len(t)],l_hip[1:len(t)]*180/np.pi,'k',label='Cadera izquierda')
    plt.legend(loc="upper left")
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Angulo [o]')
    
    plt.subplot(412)
    plt.plot(t[1:len(t)],r_knee[1:len(t)]*180/np.pi,'r',label='Rodilla derecha')
    plt.plot(t[1:len(t)],l_knee[1:len(t)]*180/np.pi,'k',label='Rodilla izquierda')
    #plt.plot(t[peaks], r_knee[peaks]*180/np.pi,marker='o')
    
    plt.legend(loc="upper left")
    plt.ylim(0, 150)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Angulo [o]')

    plt.subplot(413)
    plt.plot(t[1:len(t)],r_arm[1:len(t)]*180/np.pi,'r',label='Hombro derecho')
    plt.plot(t[1:len(t)],l_arm[1:len(t)]*180/np.pi,'k',label='Hombro izquierdo')
    plt.legend(loc="upper left")

    plt.subplot(414)
    plt.plot(t[1:len(t)],r_ankle[1:len(t)]*180/np.pi,'r',label='Tobillo derecho')
    plt.plot(t[1:len(t)],l_ankle[1:len(t)]*180/np.pi,'k',label='Tobillo izquierdo')
    plt.legend(loc="upper left")
    
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Angulo [o]')
    
    
    fig = plt.figure()
    axF = fig.add_subplot(221)
    axS = fig.add_subplot(222)
    ax = fig.add_subplot(223, projection='3d')
    ax2 = fig.add_subplot(224, projection='3d')

    
    Mins = np.min(np.min(p3ds,0),0)-100
    Maxs = np.max(np.max(p3ds,0),0)+100

    
    current_frame = 0 # frames of the video
    for framenum, kpts3d in enumerate(p3ds):
        framenum
        while current_frame<show_frame[framenum]:
            ret, frameF = capF.read()
            ret, frameS = capS.read()
            current_frame += 1
        
        axF.imshow(cv.cvtColor(frameF, cv.COLOR_BGR2RGB))
        axF.axis('off')
        axS.imshow(cv.cvtColor(frameS, cv.COLOR_BGR2RGB))
        axS.axis('off')


        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)
                ax2.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        # plot keypoints
        for i in range(Npoints):
            ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
            ax2.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
        
        #for _c in connections:
           # ax.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')
           # ax2.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')

        ####################### right plot (front)
        ax2.set_xlim3d([Mins[0], Maxs[0]])
        ax2.set_ylim3d([Mins[1], Maxs[1]])
        ax2.set_zlim3d([Mins[2], Maxs[2]])
        ax2.view_init(elev=0, azim=0)
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_zticks([])
        ####################### Left plot (side)
        ax.set_xlim3d([Mins[0], Maxs[0]])
        ax.set_ylim3d([Mins[1], Maxs[1]])
        ax.set_zlim3d([Mins[2], Maxs[2]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=0, azim=-90)
        
        plt.title(framenum)
        figname='./Frames/Fig_'+str(counter)+'.png'
        plt.savefig(figname, bbox_inches='tight')
        plt.pause(0.005)
        if framenum==show_frame[0]:
            plt.pause(10)
        ax.cla()
        ax2.cla()
        axF.cla()
        axS.cla()
        
        counter=counter+1

if __name__ == '__main__':
    Nvideo = '20'
    Date = '09_07_2026'
    Calib_n = '17'
    capF = cv.VideoCapture(r'./Videos/' + Date + '/pcte' + Nvideo + 'A.avi')
    capS = cv.VideoCapture(r'./Videos/' + Date + '/pcte' + Nvideo + 'B.avi')
    
    #npz = np.load('Floor_points.npz')
    #connections=npz['connections']
    #p3dsf=npz['p3dsf']
    npz = np.load(r'./Videos/' + Date + '/pcte' + Nvideo+'_t.npz')
    t = npz['t']
    p3ds = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    #p3dsf = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    mdic = {"body_pose": p3ds,"t":t}
    #savemat(".\pose_3d_video_"+Nvideo+".mat", mdic)
    
    visualize_3d(p3ds,capF,capS,t)
    capF.release() 
    capS.release()