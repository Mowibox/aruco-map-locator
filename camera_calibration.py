"""
    @file        camera_calibration.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Code for camera calibraiton
    @version     1.0
    @date        2024-12-25
    
"""

# Imports
import os
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
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)*SQUARE_SIZE

objpoints = [] # 3D points
imgpoints = [] # 2D points
last_detection_time = 0

app = Flask(__name__)


class CameraThread(threading.Thread):
    """
    Class to capture camera frames with multithreading
    """
    def __init__(self, camera):
        threading.Thread.__init__(self)
        self.camera = camera
        self.running = True
        self.frame = None

    def run(self):
        """
        Captures the frames
        """
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.frame = frame

    def stop(self):
        """
        Stops the camera thread
        """
        self.running = False


def parse_args():
    """
    Parses the command-line arguments
    """
    parser = argparse.ArgumentParser(description="Camera calibration")
    parser.add_argument('--max_img', type=int, default=15, help='Number of images to capture before stopping calibration.')
    args = parser.parse_args()
    return args


def save_calibration(camera_matrix, dist_coeffs):
    """
    Saves the camera calibration parameters in a yaml file

    @param: camera_matrix: The intrinsic matrix
    @param: dist_coeffs: The distortion coefficients
    """
    root = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(root, "ros2_ws", "src", "image_provider", "config")
    os.makedirs(config_dir, exist_ok=True)

    yaml_path = os.path.join(config_dir, "cam_params.yaml")

    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist()
    }
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(calibration_data, yaml_file, default_flow_style=False)
        print(f"Calibration data has been saved to {yaml_path}.\n")



def camera_calibration(MAX_IMAGES):
    """
    Performs chessboard camera calibration

    @param MAX_IMAGES: The number of images to capture before stopping calibration
    """
    global N_IMAGES, objpoints, imgpoints, last_detection_time

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
            cv2.putText(frame, n_img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (65, 55, 255), 2, cv2.LINE_AA)


            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
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
        print(f"Camera Matrix:\n{camera_matrix}\n")
        print(f"Distortion coefficients:\n{dist_coeffs}\n")

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print(f"Re-projection error:\n{mean_error/len(objpoints)}\n")

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