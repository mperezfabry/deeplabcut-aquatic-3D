import os
import yaml
import glob
import deeplabcut

# 1. Setup paths
base_path = os.path.expanduser("~/deployment_workspace/cloud_deployment")
dlc_parent = os.path.join(base_path, "fish-dlc/fish-dlc")
video_dir = os.path.join(base_path, "videos/4k_copies")

# 2. Patch YAMLs dynamically
print("🔧 Patching config.yaml paths...")
yaml_files = glob.glob(os.path.join(dlc_parent, "*/config.yaml"))
for yaml_path in yaml_files:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    
    new_project_path = os.path.dirname(yaml_path)
    if config.get('project_path') != new_project_path:
        config['project_path'] = new_project_path
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

# 3. Explicit video mapping
video_map = {
    "SandbarShark_Top": "20260121_SERt2tp_goproGX010004_trimmed.mp4",
    "SandbarShark_Left": "20260121_SERt2w1_goproGX010125_trimmed.mp4",
    "SandbarShark_Right": "20260121_SERt2w2_goproGX010153_trimmed.mp4",
    "SandbarShark_Front": "20260121_SERt2w3_akaso131835_trimmed.mp4"
}

all_videos = glob.glob(os.path.join(video_dir, "*.mp4"))

# 4. Run Analysis
for yaml_path in yaml_files:
    project_name = os.path.basename(os.path.dirname(yaml_path))
    print(f"\n--- 🦈 Analyzing Project: {project_name} ---")
    
    # Logic: Wand gets all videos. Sharks get their mapped video.
    if "Calibration_Wand" in project_name:
        target_videos = all_videos
    else:
        mapped_file = next((file for key, file in video_map.items() if key in project_name), None)
        target_videos = [os.path.join(video_dir, mapped_file)] if mapped_file else []

    if not target_videos:
        print(f"⏭️ Skipping {project_name}: No mapped video found.")
        continue

    print(f"📹 Processing {len(target_videos)} video(s)...")
    
    # Run Inference & Filtering
    deeplabcut.analyze_videos(yaml_path, target_videos, videotype='.mp4', gputouse=0, save_as_csv=True)
    deeplabcut.filterpredictions(yaml_path, target_videos, videotype='.mp4')

print("\n✅ Smart Analysis Complete.")