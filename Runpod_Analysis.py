import os
import sys
import yaml
import toml
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure DeepLabCut doesn't pre-allocate all VRAM on import
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import deeplabcut

# --- RUNPOD HARDCODED CONFIGURATION ---
WORKING_DIR = "/workspace/fish-dlc"
VIDEO_DIR = "/workspace/videos"

# Wand Timing & Geometry
WAND_START_TIME = "0:19"
WAND_END_TIME = "2:25"
FPS = 30
WAND_Z = [43.0, 190.0, 372.0, 597.0, 875.0]
WAND_X = [100.0, -100.0, 100.0, -100.0, 100.0]

def time_to_frames(t_str, fps):
    if ':' in t_str:
        m, s = map(int, t_str.split(':'))
        return (m * 60 + s) * fps
    return int(t_str) * fps

WAND_START_FRAME = time_to_frames(WAND_START_TIME, FPS)
WAND_END_FRAME = time_to_frames(WAND_END_TIME, FPS)

print(f"Starting Analysis Pipeline...")
print(f"Working Directory: {WORKING_DIR}")
print(f"Video Directory: {VIDEO_DIR}")

# --- VALIDATE PROJECTS & VIDEOS ---
if not os.path.exists(WORKING_DIR) or not os.path.exists(VIDEO_DIR):
    print("❌ ERROR: Required directories do not exist. Check your /workspace paths.")
    sys.exit(1)

project_configs = {}
found_folders = [f.path for f in os.scandir(WORKING_DIR) if f.is_dir()]
for p_path in found_folders:
    cfg_path = os.path.join(p_path, "config.yaml")
    if os.path.exists(cfg_path):
        fname = os.path.basename(p_path)
        if "Wand" in fname: project_configs['Wand'] = cfg_path
        elif "Top" in fname: project_configs['Top'] = cfg_path
        elif "Left" in fname: project_configs['Left'] = cfg_path
        elif "Right" in fname: project_configs['Right'] = cfg_path
        elif "Front" in fname: project_configs['Front'] = cfg_path

if len(project_configs) < 5:
    print("❌ ERROR: Missing trained project folders. Ensure all 5 models are in the working directory.")
    sys.exit(1)

video_files = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
video_map = {}

for v in video_files:
    v_lower = os.path.basename(v).lower()
    if "top" in v_lower or "tp" in v_lower: video_map['Top'] = v
    elif "left" in v_lower or "w1" in v_lower: video_map['Left'] = v
    elif "right" in v_lower or "w2" in v_lower: video_map['Right'] = v
    elif "front" in v_lower or "w3" in v_lower: video_map['Front'] = v

print("Videos mapped:", video_map)

# --- CONFIGURE 3D GEOMETRY ---
shark_parts = ['Snout', 'DorsalFin', 'LeftPectoral', 'RightPectoral', 'TailBase', 'TailTip']
shark_skeleton = [['Snout', 'DorsalFin'], ['DorsalFin', 'TailBase'], ['TailBase', 'TailTip'], 
                  ['LeftPectoral', 'DorsalFin'], ['RightPectoral', 'DorsalFin']]
wand_parts = [f'Ball{i+1}' for i in range(len(WAND_Z))]
wand_skeleton = [[wand_parts[i], wand_parts[i+1]] for i in range(len(wand_parts)-1)]

for name, cfg_path in project_configs.items():
    with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
    if name == 'Wand':
        cfg['bodyparts'] = wand_parts
        cfg['skeleton'] = wand_skeleton
        cfg['pcutoff'] = 0.9
    else:
        cfg['bodyparts'] = shark_parts
        cfg['skeleton'] = shark_skeleton
        cfg['pcutoff'] = 0.8
    with open(cfg_path, 'w') as f: yaml.safe_dump(cfg, f)

ball_z_mm = np.array(WAND_Z, dtype=np.float32)
ball_x_mm = np.array(WAND_X, dtype=np.float32)
constraints = []
for i in range(len(ball_z_mm) - 1): 
    dist = np.sqrt((ball_x_mm[i] - ball_x_mm[i+1])**2 + (ball_z_mm[i] - ball_z_mm[i+1])**2)
    constraints.append( (f"Ball{i+1}-Ball{i+2}", dist) )
for i in range(len(ball_z_mm) - 2): 
    dist = np.sqrt((ball_x_mm[i] - ball_x_mm[i+2])**2 + (ball_z_mm[i] - ball_z_mm[i+2])**2)
    constraints.append( (f"Ball{i+1}-Ball{i+3}", dist) )

anipose_config = {
    "project": "Shark_Tracking_Metric",
    "model_folder": WORKING_DIR,
    "video_extension": "MP4", 
    "calibration": {
        "calibration_type": "wand",
        "animal_calibration": False,
        "fisheye": False
    },
    "triangulation": {
        "cam_regex": "Cam_([A-Za-z0-9]+)", 
        "optim": True,
        "constraints": [[p[0].split('-')[0], p[0].split('-')[1], float(p[1])] for p in constraints],
        "scale_smooth": 2,
        "spatial_smooth": 2,
        "repro_error_threshold": 15
    },
    "labeling": {"scheme": [wand_parts]}
}
with open(os.path.join(WORKING_DIR, "config.toml"), "w") as f: toml.dump(anipose_config, f)
print(f"✅ Project Geometry Updated.")

# --- RUN DLC ANALYSIS ---
pose_folder = os.path.join(WORKING_DIR, "pose_2d")
os.makedirs(pose_folder, exist_ok=True)

print("\n🦈 Analyzing Sharks...")
for cam_name, video_path in video_map.items():
    cfg = project_configs[cam_name]
    print(f"   Processing {cam_name} View...")
    scorername = deeplabcut.analyze_videos(cfg, [video_path], save_as_csv=True, allow_growth=True)
    source_h5 = str(Path(video_path).with_suffix('')) + scorername + ".h5"
    dest_h5 = os.path.join(pose_folder, f"Trial1_Cam_{cam_name}.h5")
    
    if os.path.exists(source_h5):
        shutil.copy(source_h5, dest_h5)
        print(f"   ✅ Saved {cam_name} data to pose_2d.")
    else:
        print(f"   ❌ ERROR: DLC failed to generate {source_h5}.")

print("\n🪄 Tracking Calibration Wand...")
wand_cfg = project_configs['Wand']
for cam_name, video_path in video_map.items():
    print(f"   Searching for wand in {cam_name}...")
    scorername = deeplabcut.analyze_videos(wand_cfg, [video_path], save_as_csv=True, allow_growth=True)
    source_csv = str(Path(video_path).with_suffix('')) + scorername + ".csv"
    
    if not os.path.exists(source_csv):
        print(f"   ❌ ERROR: Wand analysis failed to generate {source_csv}")
        continue
        
    print(f"   🧹 Filtering noise ({WAND_START_FRAME}-{WAND_END_FRAME})...")
    df = pd.read_csv(source_csv, header=[0,1,2], index_col=0)
    mask_bad = (df.index < WAND_START_FRAME) | (df.index > WAND_END_FRAME)
    scorer = df.columns.get_level_values(0)[0]
    for bp in df.columns.get_level_values(1).unique():
        df.loc[mask_bad, (scorer, bp, 'likelihood')] = 0
        
    dest_csv = os.path.join(pose_folder, f"Calibration_Cam_{cam_name}.csv")
    df.to_csv(dest_csv)

print("\n🎬 Generating Labeled Videos for Verification...")
for cam_name, video_path in video_map.items():
    cfg = project_configs[cam_name]
    print(f"   Rendering {cam_name}...")
    deeplabcut.create_labeled_video(cfg, [video_path], draw_skeleton=True)

# --- RUN 3D CALIBRATION ---
config_path = os.path.abspath(os.path.join(WORKING_DIR, "config.toml"))
print(f"\n📐 Running Anipose Calibration...")
subprocess.run(["anipose", "calibrate", "--config", config_path], check=True)

print("🧊 Triangulating 3D Points...")
subprocess.run(["anipose", "triangulate", "--config", config_path], check=True)

print("\n🎉 Analysis & 3D Calibration Complete.")