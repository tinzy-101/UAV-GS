import cv2
import os
import numpy as np

image_dir = './video_frames'


# # open video + create capture object --> only need to run once
# video = cv2.VideoCapture('Minecraft_stitch_test.mp4')

# count, success = 0, True
# while success:
#     success, image = video.read() # read next frame
#     if success: # only process valid frames 
#         cv2.imwrite(f'{image_dir}/frame{count}.jpg', image) # save frame
#         count += 1

# # release vid capture object + free resources
# video.release() 

# load all frames in opencv
frames = []

n = 15 # select every nth image
for i, img in enumerate(os.listdir(image_dir)):
    if i % n == 0:
        img_path = os.path.join(image_dir, img)
        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            frames.append(cv_img)
        else:
            print(f"Couldn't load image {img_path}")

print(f"Successfully loaded {len(frames)} images")

# if frames:
#     cv2.imshow("First image", frames[0])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# detect features
sift = cv2.SIFT_create()
keypoints = [] # list of lists of keypoints per image, keypoints[0] = array of keypts for 1st image
descriptors = [] # list of lists of descriptors per image (one descriptor per keypoint), descriptors[0] = array of descs for 1st image
for frame in frames:
    keypoint, descriptor = sift.detectAndCompute(frame, None)
    keypoints.append(keypoint)
    descriptors.append(descriptor)
print(f"Successfully detected {len(keypoints)} keypoints and {len(descriptors)} descriptors")
assert len(keypoints) == len(descriptors), "Do not have same number keypoints and descriptors"

# these should match!
print(f"num keypts first image: {len(keypoints[0])}")
print(f"num descriptors first image: {len(descriptors[0])}")

# i, i+2
# up to i + (n-1) , i+n 
# 0, 1, 2, 3
# 4 keypoints --> n=4-2 = 2
# 0-1, 1-2, 2-3 

# find matches + compute homography matrix for each successive pair of frames 
# choose frames[(n-1)/2] for n images to be the reference image --> all other images should map to this scoordinate sys
matches_arr = [] # matches_arr[i] = list of matches for i and i+1 frames 
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
reference_img_idx = ((len(frames) - 1) / 2)
reference_img = frames[reference_img_idx]
homography_matrix_arr = []
for i in range(len(keypoints) - 2):
    # find matches w BFMatcher and sort by distance
    matches = bf.match(descriptors[i], descriptors[i+1])
    matches = sorted(matches, key=lambda x: x.distance)
    matches_arr.append(matches)
    
    # find locations of good matches
    src_pts = np.float32(keypoints[i][m.queryIdx].pt for m in matches[i]).reshape(-1, 1, 2)
    dst_pts = np.float32(keypoints[i+2][m.trainIdx].pt for m in matches[i]).reshape(-1, 1, 2)

    # compute homography (pairwise b/w successive images)
    # multiply by previous hoomography matrix to get full mapping back to reference 
    H, homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # homography_matrix_arr.append(homography) # --> only automatcically store the first one 

    # get dims of images
    h1, w1 = frames[i].shape[:2]
    h2, w2 = frames[i+1].shape[:2]

    # get dims of canvas
    pts = np.float32([[0,0], [0,h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    img2_warped = cv2.warpPerspective(frames[i+1], H, (w1+w2, h1))

    # place 1st image on canvas
    img2_warped[0:h1, 0:w1] = frames[0]

    # blend images
    result = img2_warped

    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


