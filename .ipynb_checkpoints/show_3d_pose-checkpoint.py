import numpy as np
import matplotlib.pyplot as plt
from utils import DLT, kalman_acceleration_smooth, postprocess_3d_keypoints
import cv2 as cv
from scipy.io import savemat
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D
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

def x_rotation(vector,theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    return np.dot(R,vector)

def y_rotation(vector,theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R,vector)

def z_rotation(vector,theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta), np.cos(theta),0],[0,0,1]])
    return np.dot(R,vector)

def define_steps(r_knee,t):
    peaks, properties = find_peaks(r_knee*180/np.pi, height=50, distance=25, prominence=1)
    peaks = np.insert(peaks, 0, 0)

    start_step=0
    steps_start_end=[]
    t_steps=[]
    
    for i in range(len(peaks)-2):
        for end_step in range(peaks[i+1],peaks[i+2]):
            if (r_knee[end_step]*180/np.pi)<15:
                if (t[peaks[i+1]]-t[start_step])<=1.5:
                    steps_start_end.append([start_step,end_step+1])
                    t_steps.append(t[end_step]-t[start_step])
                    #this_step = r_knee[start_step:end_step+1]*180/np.pi
                start_step=end_step+1
                break

    return steps_start_end,t_steps

def get_steps(steps_start,angles):
    angle_steps=[]
    for i in range(len(steps_start)):
        this_step = angles[steps_start[i][0]:steps_start[i][1]+1]*180/np.pi
        angle_steps.append(this_step)
    
    return angle_steps
    
def visualize_3d(p3ds,p3dsf,capF,capS,t):

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

    ## Translate skeleton to the origin of the system
    p3ds[0,:,:]=p3ds[1,:,:]

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
    
    ## commpute left knee angle
    #6-8 vs 8-10
    
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
    
    for i in range(Nframes):
        # ankle 
        l_ankle[i]=angle_between(p3ds[i,12,:]-p3ds[i,10,:], p3ds[i,8,:]-p3ds[i,10,:])-np.pi/2
        r_ankle[i]=angle_between(p3ds[i,13,:]-p3ds[i,11,:], p3ds[i,9,:]-p3ds[i,11,:])-np.pi/2
        
        # knee    
        l_knee[i]=angle_between(p3ds[i,6,:]-p3ds[i,8,:], p3ds[i,8,:]-p3ds[i,10,:])
        r_knee[i]=angle_between(p3ds[i,7,:]-p3ds[i,9,:], p3ds[i,9,:]-p3ds[i,11,:])
        # hip
        l_hip[i]=angle_between(p3ds[i,0,:]-p3ds[i,6,:], p3ds[i,6,:]-p3ds[i,8,:])
        r_hip[i]=angle_between(p3ds[i,1,:]-p3ds[i,7,:], p3ds[i,7,:]-p3ds[i,9,:])
        
        #shoulder
        l_arm[i]=angle_between(p3ds[i,1,:]-p3ds[i,7,:], p3ds[i,1,:]-p3ds[i,3,:])
        r_arm[i]=angle_between(p3ds[i,0,:]-p3ds[i,6,:], p3ds[i,0,:]-p3ds[i,2,:])
    ###### Detect steps

    r_hip = kalman_acceleration_smooth(r_hip, process_noise=1e-2, measurement_noise=1e-4)
    r_knee = kalman_acceleration_smooth(r_knee, process_noise=1e-2, measurement_noise=1e-6)
    r_ankle = kalman_acceleration_smooth(r_ankle, process_noise=1e-2, measurement_noise=1e-4)
        
    start_steps,t_steps=define_steps(r_knee,t)
    steps_r_knee=get_steps(start_steps,r_knee)
    steps_l_knee=get_steps(start_steps,l_knee)
    steps_l_hip=get_steps(start_steps,l_hip)
    steps_r_hip=get_steps(start_steps,r_hip)
    steps_l_ankle=get_steps(start_steps,l_ankle)
    steps_r_ankle=get_steps(start_steps,r_ankle)

    
    
    Nsteps=len(steps_r_knee)
    fig0 = plt.figure()

    
    for i in range(Nsteps):
       
        this_color=i/(Nsteps+3)
        x_percent = np.linspace(0, 100, len(steps_r_knee[i]))
        
        plt.subplot(231)
        plt.plot(x_percent, steps_r_knee[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo rodilla derecha [o]')

        plt.subplot(234)
        plt.plot(x_percent, steps_l_knee[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo rodilla izquierda [o]')

        plt.subplot(232)
        plt.plot(x_percent, steps_r_hip[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo cadera derecha [o]')

        plt.subplot(235)
        plt.plot(x_percent, steps_l_hip[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo cadera izquierda [o]')

        plt.subplot(233)
        plt.plot(x_percent, steps_r_ankle[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo tobillo derecha [o]')

        plt.subplot(236)
        plt.plot(x_percent, steps_l_ankle[i],color=[this_color,this_color,this_color])
        plt.xlabel('Porcentaje de marcha [%]')
        plt.ylabel('Angulo tobillo izquierda [o]')

    fig1 = plt.figure()
    for i in range(Nsteps):
        
        plt.subplot(233)
        plt.plot(1,t_steps[i],marker='o',color=[0.5,0.5,0.5])
        
    #plt.subplot(223)
    t_steps2=np.array(t_steps)
    print("Promedio duracion de paso: ", np.mean(t_steps2))
    plt.errorbar(1,np.mean(t_steps2),np.std(t_steps2),fmt='o-',color='k')
    plt.ylabel('Tiempo de cada paso [s]') 
    
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

    
    #plt.plot(t,p3ds[:,10,0],'b')
    #plt.plot(t,p3ds[:,10,1],'g')
    #plt.plot(t,p3ds[:,10,2],'r')
    plt.pause(1)
    
    
    fig = plt.figure()
    axF = fig.add_subplot(221)
    axS = fig.add_subplot(222)
    ax = fig.add_subplot(223, projection='3d')
    ax2 = fig.add_subplot(224, projection='3d')

    
    #new_z = np.zeros((2,3))
    new_z = ((p3ds[0,0,:]-p3ds[0,10,:])+(p3ds[0,1,:]-p3ds[0,11,:]))/2
    alpha_z = angle_between(new_z, np.array([0.0, 0.0, 1.0]))
    new_z = z_rotation(new_z, -alpha_z)

    
    Mins = np.min(np.min(p3ds,0),0)-100
    Maxs = np.max(np.max(p3ds,0),0)+100

    
    current_frame = 0 # frames of the video
    for framenum, kpts3d in enumerate(p3ds):
        framenum
        while current_frame<show_frame[framenum]:
            ret, frameF = capF.read()
            ret, frameS = capS.read()
            current_frame += 1
        
        #if current_frame not in show_frame: 
            #continue #skip every 2nd frame and frames that were not detected correctly by mediapipe
        
        axF.imshow(cv.cvtColor(frameF, cv.COLOR_BGR2RGB))
        axF.axis('off')
        axS.imshow(cv.cvtColor(frameS, cv.COLOR_BGR2RGB))
        axS.axis('off')

        #ax.plot(np.linspace(0,new_z[0]),np.linspace(0,new_z[1]),np.linspace(0,new_z[2]))
        #ax.plot(np.linspace(0,new_vect[0]),np.linspace(0,new_vect[1]),np.linspace(0,new_vect[2]))


        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)
                ax2.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        #uncomment these if you want scatter plot of keypoints and their indices.
        for i in range(Npoints):
            ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
            #ax2.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
            ax2.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])
        
        #for _c in connections:
           # ax.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')
           # ax2.plot(xs = [p3dsf[_c[0],0], p3dsf[_c[1],0]], ys = [p3dsf[_c[0],1], p3dsf[_c[1],1]], zs = [p3dsf[_c[0],2], p3dsf[_c[1],2]], c = 'red')

        #ax.set_axis_off()
        #ax2.set_axis_off()
        ####################### right plot (front)
        ax2.set_xlim3d([Mins[0], Maxs[0]])
        ax2.set_ylim3d([Mins[1], Maxs[1]])
        ax2.set_zlim3d([Mins[2], Maxs[2]])
        ax2.view_init(elev=0, azim=6,roll=-90)
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
        ax.view_init(elev=90, azim=-110,roll=150)
        
        plt.title(framenum)
        figname='.\Frames\Fig_'+str(counter)+'.png'
        plt.savefig(figname, bbox_inches='tight')
        plt.pause(0.05)
        if framenum==show_frame[0]:
            plt.pause(50)
        ax.cla()
        ax2.cla()
        axF.cla()
        axS.cla()
        
        counter=counter+1

if __name__ == '__main__':
    Nvideo = '18'
    capF = cv.VideoCapture(r'.\Videos\pcte'+Nvideo+'A.avi')
    capS = cv.VideoCapture(r'.\Videos\pcte'+Nvideo+'B.avi')
    
    #npz = np.load('Floor_points.npz')
    #connections=npz['connections']
    #p3dsf=npz['p3dsf']
    npz = np.load(r'.\Videos\pcte'+Nvideo+'_t.npz')
    t = npz['t']
    p3ds = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    p3dsf = read_keypoints('.\Tracking\kpts_3d_'+Nvideo+'.dat')
    mdic = {"body_pose": p3ds,"t":t}
    #savemat(".\pose_3d_video_"+Nvideo+".mat", mdic)
    
    visualize_3d(p3ds,p3dsf,capF,capS,t)
    capF.release() 
    capS.release()