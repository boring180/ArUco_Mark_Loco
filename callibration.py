import cv2 as cv
import numpy as np
import glob
import json

number_of_squares_x = 10
number_of_internal_corners_x = number_of_squares_x - 1
number_of_squares_y = 7
number_of_internal_corners_y = number_of_squares_y - 1
square_size = 0.0217 # in meters

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
objp = np.zeros((number_of_internal_corners_x * number_of_internal_corners_y,3), np.float32)
objp[:,:2] = np.mgrid[0:number_of_internal_corners_x,0:number_of_internal_corners_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('captures/*.jpg')

effective_chessboard_count = 0
total_chessboard_count = 0

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (number_of_internal_corners_x,number_of_internal_corners_y), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(f'{fname} pattern found')
        effective_chessboard_count += 1
    else:
        print(f'{fname} pattern not found')
        
    total_chessboard_count += 1

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(f'Camera matrix: {mtx}')
print(f'Distortion coefficients: {dist}')
print(f'Effective Chessboard: {effective_chessboard_count}/{total_chessboard_count}')

with open('calibration.json', 'w') as f:
    json.dump({'mtx': mtx.tolist(), 'dist': dist.tolist()}, f)