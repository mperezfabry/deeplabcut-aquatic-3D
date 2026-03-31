import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to one of your extracted 4K frames
img_path = '/mnt/Data/Projects/cloud_deployment/Scripts/Cloud_Deployment_SER-Dev-2026-03-28/labeled-data/20260318SER_t7w1_redlinear_GX010260/img0373.png'

# Load and display the image
img = mpimg.imread(img_path)
fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(img)
ax.set_title("1. Click TOP-LEFT of GoPro  |  2. Click BOTTOM-RIGHT of GoPro")

# Wait for exactly 2 clicks
print("Waiting for your clicks on the image...")
points = plt.ginput(2, timeout=0)
plt.close()

if len(points) == 2:
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    # Calculate the FFMPEG parameters
    x = int(min(x1, x2))
    y = int(min(y1, y2))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    
    print(f"\n✅ Target acquired: x={x}, y={y}, width={w}, height={h}\n")
    print("Run this exact command in your terminal:\n")
    print(f"ffmpeg -i /mnt/Data/Projects/cloud_deployment/videos/20260318_SERedt/20260318SER_t7w1_redlinear_GX010260.MP4 -vf \"drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill\" -c:a copy /mnt/Data/Projects/cloud_deployment/videos/20260318_SERedt/20260318SER_t7w1_redlinear_GX010260_MASKED.MP4")
else:
    print("Coordinates not captured. Please run the cell again and click twice.")