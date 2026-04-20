# ==============================================================================
# EAGLE RAY MASTER PIPELINE: INFERENCE -> SYNC -> IMPUTATION -> 3D
# ==============================================================================
import os
import glob
import re
import tkinter as tk
from tkinter import filedialog
import deeplabcut
import pandas as pd
import numpy as np

# --- 1. HARDCODED CONFIGURATIONS ---
DLC_CONFIGS = {
    'tp': '/mnt/Data/Projects/cloud_deployment/AprilProjects/EagleRay_Top-Dev-2026-04-01/config.yaml',
    'w1': '/mnt/Data/Projects/cloud_deployment/AprilProjects/EagleRay_W1-Dev-2026-04-01/config.yaml',
    'w2': '/mnt/Data/Projects/cloud_deployment/AprilProjects/EagleRay_W2-Dev-2026-04-01/config.yaml',
    'w3': '/mnt/Data/Projects/cloud_deployment/AprilProjects/EagleRay_W3-Dev-2026-04-01/config.yaml'
}

# DEVS: Paste the output of the multiview_ball_calib library here.
CAMERA_MATRICES = {
    'tp': np.array([[ 1.20000000e+03,  0.00000000e+00,  9.60000000e+02,  0.00000000e+00],
                    [ 0.00000000e+00,  1.20000000e+03,  5.40000000e+02,  0.00000000e+00],
                    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00]]),
    
    'w1': np.array([[ 5.56732543e+01, -1.48546877e+03,  3.89721726e+02,  2.11535589e+05],
                    [-1.11750718e+03, -2.17756031e+02,  6.59818175e+02, -2.49113264e+06],
                    [-5.44731866e-01, -7.59795464e-01, -3.54933862e-01,  4.12945703e+03]]),
    
    'w2': np.array([[ 6.93758615e+02,  1.04370424e+03,  8.89370815e+02, -2.51675924e+06],
                    [-9.30331386e+02,  3.98416896e+02,  8.41039528e+02, -4.59641266e+06],
                    [ 2.58964828e-01, -7.30389000e-02,  9.63121247e-01, -3.71460773e+03]]),
    
    'w3': np.array([[-1.37425874e+03,  6.70423747e+02,  1.53443556e+02,  1.80733235e+06],
                    [ 2.40556609e+02,  1.11692156e+03,  6.52854316e+02, -8.48444536e+05],
                    [-3.47913160e-01,  8.21968583e-01, -4.50914716e-01,  9.33544415e+03]])
}

BODYPARTS = ['Snout', 'LeftWingtip', 'RightWingtip', 'TailBase']

# --- 2. GUI & TRIAL SORTING ---
def group_videos_by_trial():
    """Groups selected videos into discrete trials based on naming conventions."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print("Waiting for user to select Trial Videos...")
    videos = filedialog.askopenfilenames(title="Select all Trial Videos", filetypes=[("MP4 Videos", "*.MP4 *.mp4")])
    if not videos: exit()

    trials = {}
    for vid in videos:
        name = os.path.basename(vid)
        # Extracts everything before 'tp', 'w1', 'w2', 'w3' (e.g., 'Fem1_OFt1arrival')
        match = re.search(r'(.*?)(tp|w1|w2|w3)', name)
        if match:
            trial_base = match.group(1)
            cam_view = match.group(2)
            if trial_base not in trials: trials[trial_base] = {}
            trials[trial_base][cam_view] = vid
            
    return trials

# --- 3. THE 3D MATH ENGINE ---
def triangulate_point(P_mats, points_2d):
    """Calculates 3D coordinates using available cameras (Needs >= 2)."""
    A = []
    for P, (x, y) in zip(P_mats, points_2d):
        if np.isnan(x) or np.isnan(y): continue
        A.append(x * P[2, :] - P[0, :])
        A.append(y * P[2, :] - P[1, :])
    
    if len(A) < 4: return np.array([np.nan, np.nan, np.nan])
        
    _, _, Vh = np.linalg.svd(np.array(A))
    X = Vh[-1, :]
    return X[:3] / X[3]

# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    trials = group_videos_by_trial()
    print(f"🔍 Discovered {len(trials)} unique trials to process.")
    
    for trial_name, cam_videos in trials.items():
        print(f"\n==========================================")
        print(f"🚀 PROCESSING TRIAL: {trial_name}")
        print(f"==========================================")
        
        # STEP A: RUN NEURAL NETWORKS
        for cam, vid_path in cam_videos.items():
            if not glob.glob(vid_path.replace('.MP4', 'DLC*.csv')):
                print(f"🧠 Running ResNet-50 for {cam.upper()} view...")
                deeplabcut.analyze_videos(DLC_CONFIGS[cam], [vid_path], save_as_csv=True)
            else:
                print(f"✅ Neural Net already run for {cam.upper()}")

        # STEP B: LOAD & CLEAN DATA
        dfs = {}
        print("🧹 Cleaning data and interpolating missing labels...")
        for cam, vid_path in cam_videos.items():
            csv_file = sorted(glob.glob(vid_path.replace('.MP4', 'DLC*.csv')))[-1]
            df = pd.read_csv(csv_file, header=[0, 1, 2], index_col=0)
            scorer = df.columns.get_level_values(0)[0]
            
            # Confidence Purge & Linear Imputation
            for bp in BODYPARTS:
                bad_conf = df[scorer, bp, 'likelihood'] < 0.6
                df.loc[bad_conf, (scorer, bp, 'x')] = np.nan
                df.loc[bad_conf, (scorer, bp, 'y')] = np.nan
                
                # Interpolate (Impute) gaps of up to 30 frames automatically
                df[(scorer, bp, 'x')] = df[(scorer, bp, 'x')].interpolate(method='linear', limit=30)
                df[(scorer, bp, 'y')] = df[(scorer, bp, 'y')].interpolate(method='linear', limit=30)
                
            dfs[cam] = df[scorer]

        # STEP C: WINGBEAT SYNCHRONIZATION
        print("⏱️ Synchronizing temporal offsets...")
        # TODO: PASTE YOUR WINGBEAT CROSS-CORRELATION SCRIPT HERE.
        # It should calculate the lag and output a dictionary like this:
        offsets = {'tp': 0, 'w1': 0, 'w2': 0, 'w3': 0} # Replace with algorithm output
        
        for cam in dfs.keys():
            dfs[cam].index = dfs[cam].index - offsets[cam]
            
        # STEP D: 3D TRIANGULATION
        valid_indices = sorted(list(set.intersection(*(set(d.index) for d in dfs.values()))))
        print(f"📐 Triangulating {len(valid_indices)} synchronized 3D frames...")
        
        results_3d = []
        for frame_idx in valid_indices:
            frame_data = {'frame': frame_idx}
            for bp in BODYPARTS:
                valid_P = []
                valid_pts = []
                for cam in dfs.keys():
                    if frame_idx in dfs[cam].index:
                        x = dfs[cam].loc[frame_idx, (bp, 'x')]
                        y = dfs[cam].loc[frame_idx, (bp, 'y')]
                        if not np.isnan(x) and not np.isnan(y):
                            valid_P.append(CAMERA_MATRICES[cam])
                            valid_pts.append((x, y))
                
                pt_3d = triangulate_point(valid_P, valid_pts)
                frame_data[f'{bp}_X'] = pt_3d[0]
                frame_data[f'{bp}_Y'] = pt_3d[1]
                frame_data[f'{bp}_Z'] = pt_3d[2]
                
            results_3d.append(frame_data)
            
        # STEP E: SAVE OUTPUT
        df_3d = pd.DataFrame(results_3d).set_index('frame')
        out_path = os.path.join(os.path.dirname(cam_videos['tp']), f"{trial_name}_3D_Kinematics.csv")
        df_3d.to_csv(out_path)
        print(f"🎉 Trial Complete! 3D trajectory saved to {os.path.basename(out_path)}")

    print("\n✅ ALL TRIALS PROCESSED.")