import os
import sys
import yaml
import toml
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Ensure DeepLabCut doesn't pre-allocate all VRAM on import
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import deeplabcut

# --- GUI SETUP ---
def browse_folder(entry_field, title):
    folder = filedialog.askdirectory(initialdir=os.getcwd(), title=title)
    if folder:
        entry_field.delete(0, tk.END)
        entry_field.insert(0, folder)

def submit():
    global WORKING_DIR, VIDEO_DIR, WAND_START_FRAME, WAND_END_FRAME, FPS, WAND_Z, WAND_X
    WORKING_DIR = entry_proj.get()
    VIDEO_DIR = entry_vid.get()
    
    if not WORKING_DIR or not VIDEO_DIR:
        messagebox.showerror("Error", "Please select both folders.")
        return
    
    try:
        FPS = int(entry_fps.get())
        
        def time_to_frames(t_str, fps):
            if ':' in t_str:
                m, s = map(int, t_str.split(':'))
                return (m * 60 + s) * fps
            return int(t_str) * fps

        WAND_START_FRAME = time_to_frames(entry_start.get(), FPS)
        WAND_END_FRAME = time_to_frames(entry_end.get(), FPS)
        WAND_Z = [float(i.strip()) for i in entry_z.get().split(',')]
        WAND_X = [float(i.strip()) for i in entry_x.get().split(',')]
        
        if len(WAND_Z) != len(WAND_X) or len(WAND_Z) < 2:
             messagebox.showerror("Error", "Wand Z and X must have the same number of points.")
             return

        root.destroy()
    except ValueError:
        messagebox.showerror("Error", "Invalid Input. Check Time formats, FPS, and Wand Coordinates.")

root = tk.Tk()
root.title("Shark Tracking - Analysis Mode")
root.attributes('-topmost', True)

tk.Label(root, text="Project Folder (contains fish-dlc):").grid(row=0, column=0, sticky='w', padx=10, pady=5)
entry_proj = tk.Entry(root, width=50)
entry_proj.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_folder(entry_proj, "Select Project Folder")).grid(row=0, column=2, padx=10)

tk.Label(root, text="Video Folder (contains mp4s):").grid(row=1, column=0, sticky='w', padx=10, pady=5)
entry_vid = tk.Entry(root, width=50)
entry_vid.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: browse_folder(entry_vid, "Select Video Folder")).grid(row=1, column=2, padx=10)

tk.Label(root, text="Wand Start Time (MM:SS):").grid(row=2, column=0, sticky='w', padx=10, pady=5)
entry_start = tk.Entry(root, width=10)
entry_start.insert(0, "0:19")
entry_start.grid(row=2, column=1, sticky='w', padx=10)

tk.Label(root, text="Wand End Time (MM:SS):").grid(row=3, column=0, sticky='w', padx=10, pady=5)
entry_end = tk.Entry(root, width=10)
entry_end.insert(0, "2:25")
entry_end.grid(row=3, column=1, sticky='w', padx=10)

tk.Label(root, text="FPS:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
entry_fps = tk.Entry(root, width=10)
entry_fps.insert(0, "30")
entry_fps.grid(row=4, column=1, sticky='w', padx=10)

tk.Label(root, text="Wand Z-Coords (mm):").grid(row=5, column=0, sticky='w', padx=10, pady=5)
entry_z = tk.Entry(root, width=50)
entry_z.insert(0, "43, 190, 372, 597, 875")
entry_z.grid(row=5, column=1, padx=10)

tk.Label(root, text="Wand X-Coords (mm):").grid(row=6, column=0, sticky='w', padx=10, pady=5)
entry_x = tk.Entry(root, width=50)
entry_x.insert(0, "100, -100, 100, -100, 100")
entry_x.grid(row=6, column=1, padx=10)

tk.Button(root, text="START ANALYSIS", bg="green", fg="white", command=submit).grid(row=7, column=1, pady=20)
print("Waiting for GUI input...")
root.mainloop()

# --- VALIDATE PROJECTS & VIDEOS ---
print(f"\nScanning {WORKING_DIR} for trained models...")
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

print(f"Scanning {VIDEO_DIR} for MP4s...")
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