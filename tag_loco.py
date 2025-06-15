import cv2 as cv
import numpy as np
import json
import time

with open('calibration.json', 'r') as f:
    calibration = json.load(f)

mtx = np.array(calibration['mtx'])
dist = np.array(calibration['dist'])

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
length_of_marker = 0.094 # in meters

# Define the marker coordinate system
marker_points = np.array([
    [0, 0, 0],
    [length_of_marker, 0, 0],
    [length_of_marker, length_of_marker, 0],
    [0, length_of_marker, 0]
], dtype=np.float32)

camera = cv.VideoCapture(1)

while True:
    ret, frame = camera.read()
    if not ret:
        break
        
    corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        print(f'Found {len(ids)} markers')
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Process each detected marker
        for i in range(len(ids)):
            # Get the corners of the current marker
            marker_corners = corners[i][0]
            
            # Calculate pose
            ret, rvec, tvec = cv.solvePnP(
                marker_points,
                marker_corners,
                mtx,
                dist,
                flags=cv.SOLVEPNP_IPPE_SQUARE
            )
            
            if ret:
                # Draw coordinate axes
                cv.drawFrameAxes(frame, mtx, dist, rvec, tvec, length_of_marker/2)
                
                # Print position and rotation
                print(f"Marker {ids[i][0]}:")
                print(f"Position (x,y,z): {tvec.flatten()}")
                print(f"Rotation (rx,ry,rz): {rvec.flatten()}")
                print("-------------------")
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(0.1)

camera.release()
cv.destroyAllWindows()