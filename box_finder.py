import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.image as mpimg
import sys

# Pass the image path as an argument, or default to a known frame
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = '/mnt/Data/Projects/cloud_deployment/Scripts/Cloud_Deployment_SER-Dev-2026-03-28/labeled-data/20260318SER_t7w1_redlinear_GX010260/img0373.png'

try:
    img = mpimg.imread(img_path)
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(img)
ax.set_title("1. Select Magnifying Glass to zoom/pan.\n2. Deselect Magnifying Glass.\n3. Drag to draw box (adjust corners as needed).")

def on_select(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    x, y = int(min(x1, x2)), int(min(y1, y2))
    w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
    
    print(f"\n✅ Box Placed -> x: {x}, y: {y}, w: {w}, h: {h}")
    print(f"FFMPEG String -> drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill")

# interactive=True allows you to drag the corners of the box after placing it
rs = RectangleSelector(ax, on_select, useblit=True, interactive=True)

plt.show()