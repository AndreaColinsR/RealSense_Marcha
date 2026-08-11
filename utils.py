import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter,rts_smoother
from scipy.signal import find_peaks


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


def _mad_outlier_mask(x, thresh=3.5):
    """Return boolean mask of non-outliers using median absolute deviation."""
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.ones_like(x, dtype=bool)
    modified_z = 0.6745 * (x - med) / mad
    return np.abs(modified_z) < thresh


def define_steps_middle(r_knee, t,
                  min_peak_height=40,     # deg, minimum flexion to count as a swing peak
                  min_step_time=0.4,      # s, reject steps shorter than this
                  max_step_time=2.5,      # s, reject steps longer than this
                  angle_threshold=15,     # deg, sanity band around the expected event
                  confirm_samples=3,      # samples velocity must stay >=0 after crossing
                  outlier_mad_thresh=4, # MAD threshold to flag non-step segments (e.g. turns)
                  exclude_windows=None):  # optional list of (t_start, t_end) to force-exclude
    """
    Detect gait cycles ("steps") from a knee flexion angle signal.

    Returns
    -------
    steps_start_end : list of [start_idx, end_idx]
    t_steps : list of step durations (s)
    """

    angle_deg = r_knee * 180 / np.pi
    vel_deg = np.gradient(angle_deg, t)

    dt = np.median(np.diff(t))
    fs = 1 / dt
    min_distance = max(1, int(min_step_time * fs))

    peaks, _ = find_peaks(angle_deg, height=min_peak_height,
                           distance=min_distance, prominence=5)
    if len(peaks) < 2:
        return [], []

    # --- 1. find candidate gait events (extension->flexion reversal) between peaks ---
    events = []
    for i in range(len(peaks) - 1):
        w0, w1 = peaks[i], peaks[i + 1]
        found = None
        for idx in range(w0 + 1, w1):
            if vel_deg[idx - 1] < 0 and vel_deg[idx] >= 0:
                # debounce: velocity must stay non-negative for a few samples
                window_end = min(idx + confirm_samples, len(vel_deg))
                if np.mean(vel_deg[idx:window_end] >= 0) >= 0.8:
                    if angle_deg[idx] < angle_threshold + 10:
                        found = idx
                        break
        if found is None:
            candidate = w0 + np.argmin(angle_deg[w0:w1])
            if angle_deg[candidate] < angle_threshold:
                found = candidate
        if found is not None:
            events.append(found)

    if len(events) < 2:
        return [], []

    # --- 2. build steps only between two consecutive DETECTED events ---
    # this naturally drops the incomplete leading segment (recording start -> first event)
    # and the incomplete trailing segment (last event -> recording end)
    steps_start_end, t_steps, roms = [], [], []
    for i in range(len(events) - 1):
        start_step, end_step = events[i], events[i + 1]
        duration = t[end_step] - t[start_step]

        if not (min_step_time <= duration <= max_step_time):
            continue  # too short/long to be a normal step

        if exclude_windows:
            ts, te = t[start_step], t[end_step]
            if any(not (te < ws or ts > we) for ws, we in exclude_windows):
                continue  # overlaps a known non-step segment (e.g., turn-around)

        steps_start_end.append([start_step, end_step + 1])
        t_steps.append(duration)
        roms.append(np.max(angle_deg[start_step:end_step + 1]))

    # --- 3. automatic outlier rejection (catches turn-arounds / bad segments) ---
    if len(t_steps) >= 4:  # need enough steps for meaningful statistics
        keep = _mad_outlier_mask(t_steps, outlier_mad_thresh) & \
               _mad_outlier_mask(roms, outlier_mad_thresh)
        steps_start_end = [s for s, k in zip(steps_start_end, keep) if k]
        t_steps = [d for d, k in zip(t_steps, keep) if k]

    return steps_start_end, t_steps

def define_steps(r_knee, t,
                  min_peak_height=50,     # deg, minimum knee flexion to count as a swing peak
                  min_step_time=0.4,      # s, reject steps shorter than this
                  max_step_time=2.0,      # s, reject steps longer than this
                  angle_threshold=15):    # deg, sanity-check band around the expected event

    angle_deg = r_knee * 180 / np.pi
    vel_deg = np.gradient(angle_deg, t)   # angular velocity, deg/s

    dt = np.median(np.diff(t))
    fs = 1 / dt
    min_distance = max(1, int(min_step_time * fs))

    peaks, _ = find_peaks(angle_deg, height=min_peak_height,
                           distance=min_distance, prominence=5)

    if len(peaks) < 2:
        return [], []

    if peaks[0] != 0:
        peaks = np.insert(peaks, 0, 0)

    steps_start_end, t_steps = [], []
    start_step = 0

    for i in range(len(peaks) - 1):
        w0, w1 = peaks[i], peaks[i + 1]

        # look for the point after the flexion peak where the knee
        # stops extending and begins to flex again (vel: - -> +)
        end_step = None
        for idx in range(w0 + 1, w1):
            if vel_deg[idx - 1] < 0 and vel_deg[idx] >= 0:
                # candidate local minimum of knee angle
                if angle_deg[idx] < angle_threshold + 10:  # loose sanity band
                    end_step = idx
                    break

        if end_step is None:
            # fallback: no clean velocity zero-crossing found,
            # use the minimum-angle sample in the window instead
            candidate = w0 + np.argmin(angle_deg[w0:w1])
            if angle_deg[candidate] < angle_threshold:
                end_step = candidate
            else:
                continue

        duration = t[end_step] - t[start_step]
        if min_step_time <= duration <= max_step_time:
            steps_start_end.append([start_step, end_step + 1])
            t_steps.append(duration)

        start_step = end_step + 1

    return steps_start_end, t_steps

def fit_plane(points):
    """
    Fit a plane to a set of 3D points via least-squares (SVD).

    Returns
    -------
    centroid : (3,) a point on the plane (mean of the input points)
    normal   : (3,) unit vector orthogonal to the best-fit plane
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD of the centered points: the singular vector associated with the
    # smallest singular value points in the direction of least variance
    # in the point cloud -> that's the plane's normal
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)

    return centroid, normal

def transform_to_custom_frame(points, centroid, normal, x_from, x_to, origin=None):
    """
    Transform 3D points into a coordinate system where:
      - X axis points from `x_from` to `x_to` (e.g. point 0 -> point 3)
      - Y axis lies in-plane, orthogonal to X (Z cross X)
      - Z axis is aligned to the plane's normal (orthogonal to the plane)

    Parameters
    ----------
    points : (N, 3) array of points to transform
    centroid, normal : output of fit_plane()
    x_from, x_to : (3,) points defining the desired X direction
        (e.g. D3Points[0], D3Points[3])
    origin : (3,) or None
        Point to use as the new origin. Defaults to `centroid`.

    Returns
    -------
    transformed : (N, 3) points in the new frame
    R : (3, 3) rotation matrix used (world_to_new)
    origin : the origin of the new frame, in original coordinates
    """
    points = np.asarray(points, dtype=float)
    origin = centroid if origin is None else np.asarray(origin, dtype=float)

    z_dir = normal / np.linalg.norm(normal)

    x_dir = np.asarray(x_to, dtype=float) - np.asarray(x_from, dtype=float)
    # project out any component along the normal, so X is guaranteed to
    # lie exactly in-plane (and thus exactly orthogonal to Z) even if
    # points 0 and 3 don't sit perfectly on the fitted plane
    x_dir = x_dir - np.dot(x_dir, z_dir) * z_dir
    x_dir /= np.linalg.norm(x_dir)

    y_dir = np.cross(z_dir, x_dir)
    y_dir /= np.linalg.norm(y_dir)

    R = np.array([x_dir, y_dir, z_dir])  # rows = new basis vectors in old coords
    transformed = (points - origin) @ R.T

    if transformed[2].mean() < 0:
        # if the mean Z is negative, flip the Z axis so that the new frame
        # is "above" the plane rather than "below" it
        R[2] *= -1
        transformed = (points - origin) @ R.T

    return transformed, R, origin