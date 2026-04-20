# ==============================================================================
# EAGLE RAY MASTER PIPELINE: DYNAMIC SYNC, CALIBRATE, & 3D KINEMATICS
# ==============================================================================
import os

# --- 0. GPU & VRAM PROTECTION ---
# This MUST be set before any other imports to prevent TensorFlow from crashing
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import glob
import re
import json
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import deeplabcut
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import savgol_filter, correlate

# --- 1. CONFIGURATIONS ---
# Current Wand Setup: Zigzag Nodes (X, Z in mm)
# Based on your provided snippet: Nodes 1, 3, 5 (Right: 100); Nodes 2, 4 (Left: -100)
WAND_NODES = {
    'Node1': (100, 43),   'Node2': (-100, 190), 
    'Node3': (100, 372),  'Node4': (-100, 597),
    'Node5': (100, 875)
}

TARGET_RES = (1920, 1080)
BODYPARTS = ['Snout', 'LeftWingtip', 'RightWingtip', 'TailBase']
SKELETON = [('Snout', 'LeftWingtip'), ('Snout', 'RightWingtip'), 
            ('LeftWingtip', 'TailBase'), ('RightWingtip', 'TailBase'),
            ('TailBase', 'Snout')]

# TODO: UPDATE THESE TO THE ACTUAL PROJECT DIRECTORIES ON THE LAB COMPUTER
RAY_CONFIGS = {
    'tp': 'C:/EagleRay_Projects/EagleRay_Top/config.yaml',
    'w1': 'C:/EagleRay_Projects/EagleRay_W1/config.yaml',
    'w2': 'C:/EagleRay_Projects/EagleRay_W2/config.yaml',
    'w3': 'C:/EagleRay_Projects/EagleRay_W3/config.yaml'
}

WAND_CONFIGS = {
    'tp': 'C:/EagleRay_Projects/Wand_Tracking/config.yaml',
    'w1': 'C:/EagleRay_Projects/Wand_Tracking/config.yaml',
    'w2': 'C:/EagleRay_Projects/Wand_Tracking/config.yaml',
    'w3': 'C:/EagleRay_Projects/Wand_Tracking/config.yaml'
}

# --- 2. VIDEO UTILITIES (AUDIT & DOWNSCALE) ---
def get_video_specs(path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json "{path}"'
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    data = json.loads(result.stdout)
    return int(data['streams'][0]['width']), int(data['streams'][0]['height'])

def ensure_1080p(vid_path):
    """Checks if video is 1080p; if not, creates a downscaled 30fps copy."""
    w, h = get_video_specs(vid_path)
    if (w, h) == TARGET_RES:
        return vid_path
    
    out_path = vid_path.replace('.MP4', '_1080p_temp.MP4').replace('.mp4', '_1080p_temp.mp4')
    if not os.path.exists(out_path):
        print(f"📉 Downscaling {os.path.basename(vid_path)} (4K -> 1080p) and forcing 30fps...")
        cmd = f'ffmpeg -i "{vid_path}" -vf scale=1920:1080 -r 30 -c:v libx264 -crf 18 -preset fast -c:a copy "{out_path}"'
        os.system(cmd)
    return out_path

def crop_video(vid_path, start, end, suffix):
    """Instantly slices video using stream copy."""
    out_path = vid_path.replace('.MP4', f'_{suffix}.MP4').replace('.mp4', f'_{suffix}.mp4')
    if not os.path.exists(out_path):
        print(f"✂️ Creating {suffix} clip for {os.path.basename(vid_path)}...")
        cmd = f'ffmpeg -hide_banner -loglevel warning -y -ss {start} -to {end} -i "{vid_path}" -c copy "{out_path}"'
        os.system(cmd)
    return out_path

# --- 3. TEMPORAL SYNC ENGINE (CROSS-CORRELATION) ---
def calculate_sync_offsets(dfs, anchor='tp'):
    """Finds frame offsets by correlating the Wand Node motion across views."""
    print(f"⏱️ Calculating temporal offsets relative to {anchor.upper()}...")
    offsets = {cam: 0 for cam in dfs.keys()}
    
    # Use Node 3 (the middle node) Y-coordinate as the sync signal
    ref_signal = dfs[anchor]['Node3_y'].fillna(method='ffill').fillna(0).values
    
    for cam, df in dfs.items():
        if cam == anchor: continue
        target_signal = df['Node3_y'].fillna(method='ffill').fillna(0).values
        
        # Cross-correlation math to find the lag (peak alignment)
        correlation = correlate(ref_signal - np.mean(ref_signal), 
                                target_signal - np.mean(target_signal), 
                                mode='full')
        lag = np.argmax(correlation) - (len(target_signal) - 1)
        offsets[cam] = lag
        print(f"   -> {cam.upper()} offset: {lag} frames")
    return offsets

# --- 4. DYNAMIC 3D CALIBRATION ENGINE ---
def get_clean_wand_data(csv_path):
    """Standardizes Wand DLC data and purges low-confidence points."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    sc = df.columns.get_level_values(0)[0]
    out = pd.DataFrame(index=df.index)
    for node in WAND_NODES.keys():
        mask = df[sc, node, 'likelihood'] >= 0.6
        out[f'{node}_x'] = df[sc, node, 'x'].where(mask, np.nan)
        out[f'{node}_y'] = df[sc, node, 'y'].where(mask, np.nan)
    return out

def get_calibration_scale(p0, p1, K, R, t):
    """Calculates the millimeter scale using the known zigzag node offsets."""
    P0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P1 = K @ np.hstack((R, t))
    
    # We use Node 1 and Node 5 for the largest possible baseline
    # p0/p1 are formatted as [Node1, Node5]
    v1 = cv2.triangulatePoints(P0, P1, p0[:len(p0)//2].T, p1[:len(p1)//2].T)
    v2 = cv2.triangulatePoints(P0, P1, p0[len(p0)//2:].T, p1[len(p1)//2:].T)
    
    # Measured distance in units
    unit_dist = np.median(np.linalg.norm((v1[:3]/v1[3]).T - (v2[:3]/v2[3]).T, axis=1))
    
    # Real physical distance between Node 1 and Node 5
    real_dist = np.sqrt((WAND_NODES['Node1'][0] - WAND_NODES['Node5'][0])**2 + 
                        (WAND_NODES['Node1'][1] - WAND_NODES['Node5'][1])**2)
    return real_dist / unit_dist

def calculate_dynamic_p_matrices(dfs):
    print("\n⚙️ Calculating dynamic 3D environment geometry...")
    # Find frames where at least Node1 and Node5 are visible across all cameras
    common_idx = sorted(list(set.intersection(*(set(df.index[df['Node1_x'].notna() & df['Node5_x'].notna()]) for df in dfs.values()))))
    
    K = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]], dtype=np.float64)
    P_mats = {'tp': K @ np.hstack((np.eye(3), np.zeros((3,1))))}

    for cam in ['w1', 'w2', 'w3']:
        # Stack Node 1 and Node 5 for Essential Matrix estimation
        p0 = np.vstack((dfs['tp'].loc[common_idx, ['Node1_x', 'Node1_y']].values, dfs['tp'].loc[common_idx, ['Node5_x', 'Node5_y']].values))
        p1 = np.vstack((dfs[cam].loc[common_idx, ['Node1_x', 'Node1_y']].values, dfs[cam].loc[common_idx, ['Node5_x', 'Node5_y']].values))
        
        E, _ = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        _, R, t, _ = cv2.recoverPose(E, p0, p1, K)
        
        scale = get_calibration_scale(p0, p1, K, R, t)
        P_mats[cam] = K @ np.hstack((R, t * scale))
        
    print("✅ 3D calibration successful.")
    return P_mats

def triangulate_point(P_mats, points_2d):
    """SVD Triangulation across all available camera views."""
    A = []
    for P, (x, y) in zip(P_mats, points_2d):
        if not np.isnan(x):
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
    if len(A) < 4: return [np.nan, np.nan, np.nan]
    _, _, Vh = np.linalg.svd(np.array(A))
    res = Vh[-1, :3] / Vh[-1, 3]
    return res.tolist()

# --- 5. KINEMATICS ENGINE ---
def run_kinematics_engine(df3d, fps=30):
    print("📊 Smoothing data and calculating physical kinematics...")
    for col in df3d.columns:
        # Interpolate and smooth to remove tracking jitter
        df3d[col] = savgol_filter(df3d[col].interpolate(limit=30).fillna(method='bfill').fillna(method='ffill'), 11, 3)
    
    # 3D Velocity (mm/s) using TailBase as anchor
    dx, dy, dz = df3d['TailBase_X'].diff(), df3d['TailBase_Y'].diff(), df3d['TailBase_Z'].diff()
    df3d['Velocity_mm_s'] = np.sqrt(dx**2 + dy**2 + dz**2) * fps
    
    # 3D Wingspan (mm)
    df3d['Wingspan_mm'] = np.sqrt(
        (df3d['LeftWingtip_X'] - df3d['RightWingtip_X'])**2 + 
        (df3d['LeftWingtip_Y'] - df3d['RightWingtip_Y'])**2 + 
        (df3d['LeftWingtip_Z'] - df3d['RightWingtip_Z'])**2
    )
    return df3d

# --- 6. MAIN EXECUTION ---
if __name__ == '__main__':
    root = tk.Tk(); root.withdraw()
    target_dir = filedialog.askdirectory(title="Select Trial Folder")
    if not target_dir: exit()
    
    # User inputs for cropping
    times = [simpledialog.askstring("Input", f"{q} (MM:SS):", initialvalue=v) for q, v in 
             [("Wand Start", "00:00"), ("Wand End", "01:30"), ("Ray Start", "00:00"), ("Ray End", "03:00")]]
    if not all(times): exit()

    all_vids = [f for f in glob.glob(os.path.join(target_dir, "*.MP4")) if 'temp' not in f and 'cropped' not in f]
    temp_files = []

    # STEP A: CALIBRATION & SYNC
    wand_dfs = {}
    for vid in [v for v in all_vids if 'calib' in v]:
        cam = next(c for c in ['tp', 'w1', 'w2', 'w3'] if c in vid)
        norm = ensure_1080p(vid); temp_files.append(norm) if norm != vid else None
        crop = crop_video(norm, times[0], times[1], "wand_crop"); temp_files.append(crop)
        
        if not glob.glob(crop.replace('.MP4', 'DLC*.csv')):
            print(f"🪄 Tracking Wand in {cam.upper()} view...")
            deeplabcut.analyze_videos(WAND_CONFIGS[cam], [crop], save_as_csv=True)
        
        csv = sorted(glob.glob(crop.replace('.MP4', 'DLC*.csv')))[-1]
        wand_dfs[cam] = get_clean_wand_data(csv)

    # Automated Temporal Sync and Dynamic Calibration
    OFFSETS = calculate_sync_offsets(wand_dfs)
    for c in wand_dfs: wand_dfs[c].index -= OFFSETS[c]
    P_MATS = calculate_dynamic_p_matrices(wand_dfs)

    # STEP B: RAY TRACKING
    ray_dfs = {}
    for vid in [v for v in all_vids if 'calib' not in v]:
        cam = next(c for c in ['tp', 'w1', 'w2', 'w3'] if c in vid)
        norm = ensure_1080p(vid); temp_files.append(norm) if norm != vid else None
        crop = crop_video(norm, times[2], times[3], "ray_crop"); temp_files.append(crop)
        
        if not glob.glob(crop.replace('.MP4', 'DLC*.csv')):
            print(f"🐟 Tracking Eagle Ray in {cam.upper()} view...")
            deeplabcut.analyze_videos(RAY_CONFIGS[cam], [crop], save_as_csv=True)
            
        csv = sorted(glob.glob(crop.replace('.MP4', 'DLC*.csv')))[-1]
        df_raw = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
        sc = df_raw.columns.get_level_values(0)[0]
        ray_dfs[cam] = df_raw[sc].copy()
        ray_dfs[cam].index -= OFFSETS[cam] # Apply sync
        
        for bp in BODYPARTS:
            m = ray_dfs[cam][bp, 'likelihood'] < 0.6
            ray_dfs[cam].loc[m, (bp, 'x')] = np.nan; ray_dfs[cam].loc[m, (bp, 'y')] = np.nan

    # STEP C: TRIANGULATION
    idx = sorted(list(set.intersection(*(set(d.index) for d in ray_dfs.values()))))
    print(f"📐 Merging into 3D ({len(idx)} frames)...")
    res_3d = []
    for i in idx:
        row = {'frame': i}
        for bp in BODYPARTS:
            p2d = [(ray_dfs[c].loc[i, (bp, 'x')], ray_dfs[c].loc[i, (bp, 'y')]) for c in ray_dfs if i in ray_dfs[c].index]
            p3d = triangulate_point([P_MATS[c] for c in ray_dfs if i in ray_dfs[c].index], p2d)
            row[f'{bp}_X'], row[f'{bp}_Y'], row[f'{bp}_Z'] = p3d
        res_3d.append(row)
        
    df3d = run_kinematics_engine(pd.DataFrame(res_3d).set_index('frame'))
    df3d.to_csv(os.path.join(target_dir, f"{os.path.basename(target_dir)}_Kinematics_Report.csv"))

    # Generate Browser-Based Interactive Viewer
    fig = go.Figure()
    for bp in BODYPARTS:
        fig.add_trace(go.Scatter3d(x=df3d[f'{bp}_X'], y=df3d[f'{bp}_Y'], z=df3d[f'{bp}_Z'], mode='lines', name=bp))
    fig.write_html(os.path.join(target_dir, "3D_Interactive_Viewer.html"))

    # Space Cleanup
    for f in temp_files:
        try: os.remove(f)
        except: pass
    print(f"✅ BATCH COMPLETE. Results in {target_dir}")