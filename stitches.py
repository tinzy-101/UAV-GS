import cv2
import os
import numpy as np

image_dir = './less_video_frames'


# open video + create capture object --> only need to run once
video = cv2.VideoCapture('Minecraft_stitch_test.mp4')

count, success = 0, True
while success:
    success, image = video.read() # read next frame
    if success: # only process valid frames 
        cv2.imwrite(f'{image_dir}/frame{count}.jpg', image) # save frame
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# release vid capture object + free resources
video.release() 

# load all frames in opencv
frames = []

n = 20 # select every nth image
for i, img in enumerate(os.listdir(image_dir)):
    if i % n == 0:
        img_path = os.path.join(image_dir, img)
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            frames.append(cv_img)
        else:
            print(f"Couldn't load image {img_path}")

print(f"Successfully loaded {len(frames)} images")

# stitcher object --> handles feature detection/mapping
stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
stitcher.setWaveCorrection(False) # memory error

# stitch images
# frames_subset = [frames[0], frames[10], frames[20]]
status, full_view = stitcher.stitch(frames)

# show results
if status == cv2.Stitcher_OK:
    cv2.imshow('Full view', full_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Error during stiching: {status}")


cv2.destroyAllWindows()