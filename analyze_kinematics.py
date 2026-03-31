import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.signal import medfilt

# --- CONFIGURATION ---
FPS = 30.0  # Update this if your cameras shoot at a different framerate
PIXEL_TO_CM = 1.0  # Optional: Conversion factor if Argus isn't outputting in real-world units
RESULTS_DIR = os.path.expanduser("~/Downloads/shark_results")
INPUT_FILE = os.path.join(RESULTS_DIR, "Final_Shark_3D_Coordinates.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "Visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📊 Loading 3D Tracking Data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print("❌ ERROR: Could not find the 3D coordinates CSV. Run Argus triangulation first.")
    exit(1)

# Ensure required columns exist (Assuming Argus outputs x, y, z)
if not all(col in df.columns for col in ['x', 'y', 'z']):
    print("❌ ERROR: CSV must contain 'x', 'y', and 'z' columns.")
    exit(1)

# --- KINEMATIC CALCULATIONS ---
print("⚙️ Calculating Kinematics (Velocity & Acceleration)...")
# Time vector
df['time_sec'] = df.index / FPS

# Apply a median filter to smooth out high-frequency tracking jitter
WINDOW = 5
df['x_smooth'] = medfilt(df['x'], WINDOW)
df['y_smooth'] = medfilt(df['y'], WINDOW)
df['z_smooth'] = medfilt(df['z'], WINDOW)

# Calculate frame-to-frame displacement (Euclidean distance)
df['dx'] = df['x_smooth'].diff()
df['dy'] = df['y_smooth'].diff()
df['dz'] = df['z_smooth'].diff()
df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2) * PIXEL_TO_CM

# Calculate Velocity and Acceleration
df['velocity'] = df['distance'] / (1.0 / FPS)
df['acceleration'] = df['velocity'].diff() / (1.0 / FPS)

# Fill NaNs created by diff()
df.fillna(0, inplace=True)

# --- VISUALIZATION 1: INTERACTIVE 3D TRAJECTORY ---
print("📈 Generating Interactive 3D Trajectory...")
fig_3d = go.Figure(data=[go.Scatter3d(
    x=df['x_smooth'],
    y=df['y_smooth'],
    z=df['z_smooth'],
    mode='lines',
    line=dict(
        width=4,
        color=df['velocity'],
        colorscale='Viridis',
        colorbar_title='Velocity'
    )
)])

fig_3d.update_layout(
    title='Shark 3D Trajectory (Color = Velocity)',
    scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Depth (Z)'),
    margin=dict(l=0, r=0, b=0, t=40)
)
html_out = os.path.join(OUTPUT_DIR, "Interactive_3D_Trajectory.html")
fig_3d.write_html(html_out)


# --- VISUALIZATION 2: VELOCITY OVER TIME ---
print("📈 Generating Velocity Profile...")
plt.figure(figsize=(12, 6))
sns.lineplot(x='time_sec', y='velocity', data=df, color='b', linewidth=1.5)
plt.title('Shark Swim Velocity Over Time', fontsize=16)
plt.xlabel('Time (Seconds)', fontsize=12)
plt.ylabel('Velocity', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
vel_out = os.path.join(OUTPUT_DIR, "Velocity_Profile.png")
plt.savefig(vel_out, dpi=300)
plt.close()


# --- VISUALIZATION 3: TOP-DOWN SPATIAL DENSITY (HEATMAP) ---
print("📈 Generating Spatial Density Heatmap...")
plt.figure(figsize=(10, 8))
sns.kdeplot(x=df['x_smooth'], y=df['y_smooth'], cmap="mako", fill=True, bw_adjust=0.5)
plt.plot(df['x_smooth'], df['y_smooth'], color='white', alpha=0.3, linewidth=0.5) # Overlay faint path
plt.title('Top-Down Spatial Density (Tank Usage)', fontsize=16)
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)
plt.tight_layout()
density_out = os.path.join(OUTPUT_DIR, "Spatial_Density_Heatmap.png")
plt.savefig(density_out, dpi=300)
plt.close()

print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")