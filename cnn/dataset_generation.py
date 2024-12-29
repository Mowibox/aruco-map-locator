"""
    @file        dataset_generation.py
    @author      Mowibox (Ousmane THIONGANE)
    @brief       Code generate dataset for the CNN model
    @version     1.0
    @date        2024-12-29
    
"""

import os
import cv2 
import random
import numpy as np
import pandas as pd
import cv2.aruco as aruco

MARKER_ID_LIST = [20, 21, 22, 23]

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()

os.makedirs('dataset/img', exist_ok=True)

filepath = 'dataset/dataset.csv'
if not os.path.exists(filepath):
    columns = ['filename']
    for marker_id in MARKER_ID_LIST:
        columns.extend([f'x{marker_id}',f'y{marker_id}'])
    pd.DataFrame(columns=columns).to_csv(filepath, index=False)

cap = cv2.VideoCapture(0)

def create_occlusion(frame: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> np.ndarray: 
    """
    Creates occlusions on detected ArUco tags

    @param frame: The input image
    @param corners: The corners of the detected aruco tags 
    @param ids: The IDs of the detected ArUco tags
    """
    for i, corner_coord in enumerate(corners):
        if ids[i] in MARKER_ID_LIST:
            if random.random() > 0.5:
                xmin, ymin = np.min(corner_coord[0], axis=0).astype(int)
                xmax, ymax = np.max(corner_coord[0], axis=0).astype(int)
                
                marker_w = xmax - xmin
                marker_h = ymax - ymin

                occlusion_w = random.uniform(0.1, 1.5)*marker_w
                occlusion_h = random.uniform(0.1, 1.5)*marker_h

                xmin, ymin = int(random.uniform(xmin, xmax-occlusion_w)), int(random.uniform(ymin, ymax-occlusion_h))
                xmax, ymax = int(xmin + occlusion_w), int(ymin + occlusion_h)

                # Random grayscaled occlusion
                c = random.randint(0,255)
                color = (c, c, c)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, -1)

    return frame


def main():
    counter = 0

    while True:
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params)
        
        if ids is not None:
            ids = ids.flatten()

            ids = np.sort(ids)
            

            if all(marker_id in ids for marker_id in MARKER_ID_LIST):
                marker_coords = {}
                
                for marker_id, corner_coords in zip(ids, corners):
                    corner_coords = np.array(corner_coords[0])
                    x = np.mean(corner_coords[:, 0])
                    y = np.mean(corner_coords[:, 1])
                    marker_coords[marker_id] = (x, y)

                data_row = {'filename': f'img_{counter}.png'}
                for marker_id in MARKER_ID_LIST:
                    if marker_id in marker_coords:
                        x,y = marker_coords[marker_id]
                        data_row[f'x{marker_id}'] = x
                        data_row[f'y{marker_id}'] = y

                pd.DataFrame([data_row]).to_csv(filepath, mode='a', header=False, index=False)
            
                filename = os.path.join('dataset/img', f"img_{counter}.png")
                cv2.imwrite(filename, create_occlusion(frame, corners, ids))

                print(f"{counter} images captured and saved")
                counter+=1
            
        cv2.imshow("Dataset generation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()