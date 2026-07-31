import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter,rts_smoother


def interpolate_zeros(series,value=0):
    """
    Replace zeros with linearly interpolated values.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    pd.Series
    """
    return (
        series.replace(value, np.nan)
              .interpolate(method='linear')
              .bfill()   # fills leading NaNs if the series starts with zeros
              .ffill()   # fills trailing NaNs if the series ends with zeros
    )

def rts_acceleration_smooth(
        signal,
        dt=1/30,
        process_noise=1e-2,
        measurement_noise=0.005):

    """
    Constant acceleration Kalman filter + RTS smoother.

    State:
        x = [position, velocity, acceleration]
    """

    signal = np.asarray(signal, dtype=float)

    # Fill missing values
    signal = (
        pd.Series(signal)
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    n = len(signal)


    # ----------------------------
    # Define Kalman model
    # ----------------------------

    kf = KalmanFilter(
        dim_x=3,
        dim_z=1
    )


    # State: position, velocity, acceleration
    kf.x = np.array([
        signal[0],
        0,
        0
    ])


    # Transition matrix
    kf.F = np.array([
        [1, dt, 0.5*dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])


    # Measurement: only position observed
    kf.H = np.array([
        [1, 0, 0]
    ])


    kf.P = np.eye(3) * 10


    kf.R = np.array([
        [measurement_noise]
    ])


    q = process_noise

    kf.Q = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2, dt],
        [dt**2/2, dt, 1]
    ]) * q


    # ----------------------------
    # Forward Kalman filter
    # ----------------------------

    means = []
    covariances = []

    for z in signal:

        kf.predict()
        kf.update([z])

        means.append(kf.x.copy())
        covariances.append(kf.P.copy())


    means = np.asarray(means)
    covariances = np.asarray(covariances)


    # ----------------------------
    # RTS smoother
    # ----------------------------

    # RTS requires one F and Q per timestep
    Fs = np.repeat(
        kf.F[np.newaxis, :, :],
        n-1,
        axis=0
    )

    Qs = np.repeat(
        kf.Q[np.newaxis, :, :],
        n-1,
        axis=0
    )


    smoothed= rts_smoother(
        means,
        covariances,
        Fs,
        Qs
    )


    return smoothed[0]


def kalman_acceleration_smooth(
        signal,
        dt=1/30,
        process_noise=1e-3,
        measurement_noise=1e-2):
    """
    Constant acceleration Kalman filter for a 1D trajectory.

    State:
        x = [position, velocity, acceleration]

    Parameters:
        signal:
            1D array of measurements

        dt:
            Time between samples (seconds).
            For MediaPipe at 30 FPS use 1/30.

        process_noise:
            How much acceleration is allowed to change.
            Larger -> follows motion more closely.
            Smaller -> smoother.

        measurement_noise:
            Expected measurement noise.
            Larger -> trusts measurements less.

    Returns:
        Filtered position trajectory
    """

    signal = np.asarray(signal, dtype=float)

    # Fill missing values
    signal = (
        pd.Series(signal)
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    kf = KalmanFilter(
        dim_x=3,
        dim_z=1
    )

    # Initial state:
    # position, velocity, acceleration
    kf.x = np.array([
        signal[0],
        0,
        0
    ], dtype=float)


    # State transition matrix
    kf.F = np.array([
        [1, dt, 0.5*dt**2],
        [0, 1, dt],
        [0, 0, 1]
    ])


    # Measurement matrix:
    # We only observe position
    kf.H = np.array([
        [1, 0, 0]
    ])


    # Initial uncertainty
    kf.P = np.eye(3) * 10


    # Measurement uncertainty
    kf.R = np.array([
        [measurement_noise]
    ])


    # Process noise
    # Models uncertainty in acceleration changes
    q = process_noise

    kf.Q = np.array([
        [dt**4/4, dt**3/2, dt**2/2],
        [dt**3/2, dt**2,   dt],
        [dt**2/2, dt,      1]
    ]) * q


    filtered = np.zeros(len(signal))


    for i, measurement in enumerate(signal):

        kf.predict()

        kf.update(
            np.array([measurement])
        )

        filtered[i] = kf.x[0]


    return filtered

def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)



def postprocess_3d_keypoints(kpts_3d, process_noise=1e-2, measurement_noise=1e-2):

    # Convert list to numpy array
    kpts_3d = np.asarray(kpts_3d)

    # Expected shape: (N_frames, Npoints, 3)
    N_frames, Npoints, _ = kpts_3d.shape

    for marker in range(Npoints):

        traj = kpts_3d[:, marker, :]   # (N_frames, 3)

        for dim in range(3):
            coord = pd.Series(traj[:, dim])

            coord = interpolate_zeros(coord,-1)
            coord = kalman_acceleration_smooth(coord)

            traj[:, dim] = coord

        kpts_3d[:, marker, :] = traj

    return kpts_3d.tolist()
