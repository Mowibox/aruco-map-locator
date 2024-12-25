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
import time
import argparse
import threading
import numpy as np
from flask import Flask, Response

CHESSBOARD_SIZE = (7, 10) # Number of inners corners (row, col)
SQUARE_SIZE = 0.025       # In meters (25 mm)
N_IMAGES = 0              # Counter for the number of chessboard detections
TIME_DELAY = 3            # Minimal time between two detections

# Frame dimensions
FRAME_WIDTH = 604
FRAME_HEIGHT = 480

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)*SQUARE_SIZE

objpoints = [] # 3D points
imgpoints = [] # 2D points
last_detection_time = 0

app = Flask(__name__)


class CameraThread(threading.Thread):
    def __init__(self, camera):
        threading.Thread.__init__(self)
        self.camera = camera
        self.running = True
        self.frame = None

    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame

    def stop(self):
        self.running = False


def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument('--max_img', type=int, default=15, help='Number of images to capture before stopping calibration.')
    args = parser.parse_args()
    return args


def save_calibration(camera_matrix, dist_coeffs):
    """
    Saves the camera calibration parameters in a YAML file

    @param: camera_matrix: The camera matrix
    @param: dist_coeffs: The distorsion coefficients
    """
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distorsion_coefficients': dist_coeffs.tolist()
    }
    with open("cam_params.yaml", "w") as yaml_file:
        yaml.dump(calibration_data, yaml_file, default_flow_style=False)
        print("Calibration data has been saved to 'cam_params.yaml'.")



def camera_calibration(MAX_IMAGES):
    """
    Performs chessboard camera calibration

    @param MAX_IMAGES: The number of images to capture before stopping calibration
    """
    global N_IMAGES, objpoints, imgpoints, last_detection_time

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        return "Error, could not open camera."
    camera_thread = CameraThread(camera)
    camera_thread.start()

    while True:
        if camera_thread.frame is not None:
            frame = camera_thread.frame
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            current_time = time.time()
            if current_time - last_detection_time > TIME_DELAY:
                ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)
                    N_IMAGES+=1
                    last_detection_time = current_time

            n_img_text = f"num_of_imgs: {N_IMAGES}/{MAX_IMAGES}"
            cv2.putText(frame, n_img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (65, 55, 255), 2, cv2.LINE_AA)


            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            time.sleep(0.1)
            if not ret:
                continue
        
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            if N_IMAGES >= MAX_IMAGES:
                break

    if len(objpoints) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distorsion coefficients:\n{dist_coeffs}")
        save_calibration(camera_matrix, dist_coeffs)
        print("Press CTRL-C to close the terminal...")

    camera.release()
    camera_thread.stop()
    camera_thread.join()

@app.route('/')
def video_feed():
    MAX_IMAGES = parse_args().max_img
    return Response(camera_calibration(MAX_IMAGES),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        exit(0)