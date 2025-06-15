import cv2 as cv
import os
from datetime import datetime

camera = cv.VideoCapture(1)

while True:
    ret, frame = camera.read()
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('s'):
        if not os.path.exists('captures'):
            os.makedirs('captures')
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'captures/frame_{timestamp}.jpg'
        
        cv.imwrite(filename, frame)

camera.release()
cv.destroyAllWindows()