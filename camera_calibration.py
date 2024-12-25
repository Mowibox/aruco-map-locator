"""
    @file        camera_calibration.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Code for camera calibraiton
    @version     1.0
    @date        2024-12-25
    
"""

# Imports
import cv2
import yaml
import numpy as np
from flask import Flask, Response

CHESSBOARD_SIZE = (7, 10) # Number of inners corners (row, col)
SQUARE_SIZE = 0.025       # In meters (25 mm)
N_IMAGES = 0              # Counter for the number of chessboards detections

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)*SQUARE_SIZE

objpoints = [] # 3D points
imgpoints = [] # 2D points

app = Flask(__name__)

def camera_calibration():
    """
    Performs chessboard camera calibration
    """
    global N_IMAGES, objpoints, imgpoints

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not camera.isOpened():
        return "Error, could not open camera."

    while True:
        ret, frame = camera.read()

        frame = cv2.flip(frame, 1)
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
            N_IMAGES+=1

        n_img_text = f"num_of_imgs: {N_IMAGES}"
        cv2.putText(frame, n_img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 55, 255), 2, cv2.LINE_AA)


        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            continue
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        if N_IMAGES >= 10:
            break

    if len(objpoints) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distorsion coefficients:\n{dist_coeffs}")
        
    else:
        print("No chessboard images have been captured for calibration.")
    camera.release()
    cv2.destroyAllWindows()

@app.route('/')
def video_feed():
    return Response(camera_calibration(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        exit(0)