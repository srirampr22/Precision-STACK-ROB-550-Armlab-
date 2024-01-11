#!/usr/bin/env python3

"""!
Class to represent the camera.
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import pickle
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from copy import deepcopy
# from imutils import grab_contours


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.GridFrameNoPoints = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        self.BlockFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720, 1280, 3)).astype(np.uint8)

        # mouse clicks & calibration variables
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.homography = np.eye(3)
        self.distortion_coefficients = np.ones(5)
        self.last_click = np.array([0, 0])
        self.last_click_world = []
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-500, 550, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(
            self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        self.update_box_counter = 0
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.block_detections = []
        self.same_block_pixel_thresh = 20
        self.bool_detect = False

        self._denoising_counter = 0
        self.last_frames = {"RGB": [], "DEPTH": []}
        self.color_dict = {
            "yellow": {
                "H_l": 16,
                "S_l": 145,
                "V_l": 158,
                "H_u": 37,
                "S_u": 255,
                "V_u": 235,
                "R": 255,
                "G": 235,
                "B": 0,
            },
            "orange": {
                "H_l": 0,
                "S_l": 135,
                "V_l": 149,
                "H_u": 15,
                "S_u": 255,
                "V_u": 255,
                "R": 255,
                "G": 79,
                "B": 0,
            },
            "green": {
                "H_l": 37,
                "S_l": 76,
                "V_l": 51,
                "H_u": 89,
                "S_u": 255,
                "V_u": 255,
                "R": 102,
                "G": 255,
                "B": 0,
            },
            "blue": {
                "H_l": 95,
                "S_l": 71,
                "V_l": 66,
                "H_u": 107,
                "S_u": 255,
                "V_u": 255,
                "R": 30,
                "G": 144,
                "B": 255,
            },
            "red": {
                "H_l": 157,
                "S_l": 50,
                "V_l": 30,
                "H_u": 179,
                "S_u": 255,
                "V_u": 255,
                "R": 255,
                "G": 40,
                "B": 0,
            },
            "purple": {
                "H_l": 108,
                "S_l": 34,
                "V_l": 66,
                "H_u": 161,
                "S_u": 255,
                "V_u": 255,
                "R": 191,
                "G": 0,
                "B": 255,
            }

        }

        # Object for apriltag poses
        self.apriltag_poses = None

        # Load previous calibration if there is one available
        try:
            with open('../config/camera_calibration.pickle', 'rb') as handle:
                camera_calibration_dict = pickle.load(handle)
                self.intrinsic_matrix = camera_calibration_dict["intrinsic_matrix"]
                self.extrinsic_matrix = camera_calibration_dict["extrinsic_matrix"]
                self.distortion_coefficients = camera_calibration_dict["distortion_coefficients"]
                self.homography = camera_calibration_dict["homography"]

            self.camera_calibration_loaded = True

        except:
            self.camera_calibration_loaded = False

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1],
                         frame.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    # Block frame
    def convertQtBlockFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.BlockFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        # print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def inverse_homo(self, points, homography):
        # Each row is 1x2; Each row is a point

        n_points = points.shape[0]
        converted_points = np.zeros((n_points, 2))
        for idx in range(n_points):

            u_prime, v_prime = points[idx, :]
            # Convert to homogeneous coordinates
            p_prime = np.array([u_prime, v_prime, 1]).reshape(3, 1)

            # Compute the inverse of the homography
            H_inv = np.linalg.inv(homography)

            # Get the original point using the inverse homography
            p = np.dot(H_inv, p_prime)

            # Convert back from homogeneous to Cartesian coordinates
            u = p[0] / p[2]
            v = p[1] / p[2]

            converted_points[idx, 0] = u
            converted_points[idx, 1] = v

        return converted_points

    def get_XYZ(self, points):

        n_points = points.shape[0]
        converted_points = np.zeros((n_points, 3))
        for idx in range(n_points):
            u, v = points[idx, :]
            # print(f'u={u}, v={v}')
            z = self.DepthFrameRaw[int(v)][int(u)]

            K_inv = np.linalg.inv(self.intrinsic_matrix)

            H_inv = np.linalg.inv(self.extrinsic_matrix)

            camera_coords = (z * (K_inv @ np.array([int(u), int(v), 1]).T))

            world_coord = H_inv @ np.array([camera_coords[0],
                                           camera_coords[1], camera_coords[2], 1]).T

            X = world_coord[0]
            Y = world_coord[1]
            Z = world_coord[2]

            converted_points[idx, 0] = X
            converted_points[idx, 1] = Y
            converted_points[idx, 2] = Z

        return converted_points

    def polygonArea(self, points):

        X = points[:, 0]
        Y = points[:, 1]

        # Initialize area
        area = 0.0

        # Calculate value of shoelace formula
        j = X.shape[0] - 1
        for i in range(0, X.shape[0]):
            area += (X[j] + X[i]) * (Y[j] - Y[i])
            j = i   # j is previous vertex to i

        # Return absolute value
        return int(abs(area / 2.0))

    def homo(self, points):

        # points = [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]

        transformed_points = []

        for (u, v) in points:
            # Convert point to homogeneous coordinates
            p = np.array([[u], [v], [1]])

            # Apply homography
            p_prime = np.dot(self.homography, p)

            # Convert back to inhomogeneous coordinates
            u_prime = p_prime[0] / p_prime[2]
            v_prime = p_prime[1] / p_prime[2]

            transformed_points.append((u_prime[0], v_prime[0]))

        return transformed_points

    def is_point_inside_bounding_box(self, clicked_point, bbx_points):
        # Assuming the points are ordered in either a clockwise or counter-clockwise manner.
        a, b = clicked_point

        (u1, v1), (u2, v2), (u3, v3), (u4, v4) = bbx_points

        # Check if point is within the bounds
        if min(u1, u2, u3, u4) <= a <= max(u1, u2, u3, u4) and \
                min(v1, v2, v3, v4) <= b <= max(v1, v2, v3, v4):
            return True
        else:
            return False

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        self._denoising_counter += 1
        avg_number = 50

        raw_image = self.GridFrameNoPoints.copy()
        raw_image_to_show = self.GridFrameNoPoints.copy() 
        raw_depth = self.DepthFrameRGB.copy()

        robot_points = [(531.4611627289269, 381.2844155483755), (747.544157852484, 383.5262888033949),
                        (747.8544224174639, 657.0939374992091), (532.7893502251081, 657.8034841043298)]

        # left_bar_points = [(221,238), (238,210), (275,476), (259,479)]
        # right_bar_points = [(1151,160), (1170,174), (1150,444), (1131,442)]

        # print(self.homo(right_bar_points))

        Lbar_points = [(83.76929166319383, 221.93838261574845), (108.14325912445071, 194.71471677778587),
                       (109.16534137470619, 476.88996173700565), (88.82434848858108, 479.51984453799014)]
        Rbar_points = [(1176.0601479834515, 182.41341724442094), (1199.5462537286962, 197.03070088452628),
                       (1199.8966370886312, 481.4090479388636), (1175.9491611336316, 478.28142809445245)]

        robot_points_int = [(int(u), int(v))
                            for u, v in robot_points]
        Lbar_points_int = [(int(u), int(v))
                           for u, v in Lbar_points]
        Rbar_points_int = [(int(u), int(v))
                           for u, v in Rbar_points]

        us, vs = zip(*robot_points_int)
        ul, vl = zip(*Lbar_points_int)
        ur, vr = zip(*Rbar_points_int)

        Robot_min_u, Robot_max_u = min(us), max(us)
        Robot_min_v, Robot_max_v = min(vs), max(vs)

        L_min_u, L_max_u = min(ul), max(ul)
        L_min_v, L_max_v = min(vl), max(vl)

        R_min_u, R_max_u = min(ur), max(ur)
        R_min_v, R_max_v = min(vr), max(vr)

        raw_image[Robot_min_v:Robot_max_v,
                  Robot_min_u: Robot_max_u] = [0, 0, 0]
        raw_image[L_min_v:L_max_v, L_min_u: L_max_u] = [0, 0, 0]
        raw_image[R_min_v:R_max_v, R_min_u: R_max_u] = [0, 0, 0]

        # Blur DEPH too????

        hsv_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HSV)

        # Push the frames onto the array
        # self.last_frames["RGB"].append(blur_img)
        self.last_frames["RGB"].append(hsv_img)
        self.last_frames["DEPTH"].append(raw_depth)

        # Then check to see if we need to remove any frames from the start of the array
        if len(self.last_frames["RGB"]) > avg_number:
            self.last_frames["RGB"].pop(0)
            self.last_frames["DEPTH"].pop(0)
        elif len(self.last_frames["RGB"]) < avg_number:
            # In the case when we have less frames than our desired average
            avg_number = len(self.last_frames["RGB"])

        avg_image = self.last_frames["RGB"][0]
        avg_depth = self.last_frames["DEPTH"][0]
        for i in range(len(self.last_frames["RGB"])):
            if i == 0:
                pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(
                self.last_frames["RGB"][i], alpha, avg_image, beta, 0.0)
            avg_depth = cv2.addWeighted(
                self.last_frames["DEPTH"][i], alpha, avg_depth, beta, 0.0)

        # Then generate average
        # avg_img = np.zeros((720, 1280, 3))
        # avg_depth = np.zeros((720, 1280, 3))
        avg_image = cv2.GaussianBlur(avg_image, (5, 5), 0)

        hsv_img = avg_image
        raw_image = avg_image

        # Clean up current block detections
        self.clean_up_block_detections()
        self.update_box_counter += 1

        for color in self.color_dict.keys():

            # print(f"Searching for {color} contours. // ", end="")

            # Get the tag color
            HSV_tag_color = np.array(
                [self.color_dict[color]["R"], self.color_dict[color]["G"], self.color_dict[color]["B"]])
            color_hsv = tuple(HSV_tag_color)
            color_int = tuple(map(int, color_hsv))

            # Get lower/upper bounds for the specific color
            hsv_lowerbound = np.array(
                [self.color_dict[color]["H_l"], self.color_dict[color]["S_l"], self.color_dict[color]["V_l"]])
            hsv_upperbound = np.array(
                [self.color_dict[color]["H_u"], self.color_dict[color]["S_u"], self.color_dict[color]["V_u"]])

            # Create mask based on these bounds
            hsv_mask = cv2.inRange(hsv_img, hsv_lowerbound, hsv_upperbound)

            depth_masked = cv2.bitwise_and(raw_depth, raw_depth, mask=hsv_mask)

            t_lower = 50  # Lower Threshold
            t_upper = 150  # Upper threshold

            blur = cv2.GaussianBlur(hsv_mask, (3, 3), 0.1)

            kernel = np.ones((4, 4), np.uint8)
            hsv_mask = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)

            # Generate contours in this masked image
            current_contours, _ = cv2.findContours(
                hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # print(f"# Unfiltered ({len(current_contours)})  //  ", end="")

            num_valid = 0
            num_parallelograms = 0
            num_circles = 0
            num_undefined = 0

            # Then iterate over contours and generate writing
            for contour in current_contours:

                # Calculate contour area
                contour_area = cv2.contourArea(contour)

                # Hard check to see if area is significant enough, check threshold later
                if contour_area > 300:
                    is_parallelogram, is_square, is_rectangle, approx = self.is_rectangle(
                        contour, contour_area)
                    is_circle = self.is_circular(contour)

                    # Draw rectangle
                    if is_parallelogram:
                        num_parallelograms += 1
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        clicked_point = self.last_click

                        if (self.last_click is not None):
                            if (self.is_point_inside_bounding_box(clicked_point, box) == True):
                                self.bool_detect = True
                            else:
                                self.bool_detect = False

                        num_valid += 1

                        M = cv2.moments(approx)

                        # draw centroids
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            centroid_prime = np.array(
                                (int(cx), int(cy))).reshape((1, 2))
                            prime_points = np.concatenate(
                                (box, centroid_prime), axis=0)
                            camera_points = self.inverse_homo(
                                prime_points, self.homography)

                            # true_centroid = (int(u_t), int(v_t))
                            world_points_array = self.get_XYZ(camera_points)
                            # world_points = (X,Y,Z)

                            _corners = world_points_array[:5, :]
                            top_face_area = self.polygonArea(_corners)

                            # print(f'top aera = {top_face_area}')
                            z_block = round(world_points_array[4, 2], 2)

                            if top_face_area >= 900: # and z_block >= 30:
                                size_type = 'big'
                                size_num = 1
                            else:
                                size_type = 'small'
                                size_num = 0


                            block_info = {
                                'label': num_valid,
                                'color': str(color),
                                'color_int': color_int,
                                'centroid': prime_points[4, :],
                                'world_points': world_points_array[4, :],
                                'box': prime_points[:4, :],
                                'orientation': self.calculate_block_orientation(box=prime_points[:4, :], centroid=prime_points[4, :]),
                                'prev_orientation': None,
                                'size': size_type,    # big or small
                                'size_num': size_num,
                                'frames_since_last_detection': 0, 
                                'confidence': 1,
                                'tfa': top_face_area
                            }

                            # Check to see if block is already in our detection
                            block_idx = self.is_already_detected(block_info) 

                            # If the block is not detected, add it to our list
                            if block_idx == -1: 
                                self.block_detections.append(block_info)

                            # Otherwise, then this block is already in the list
                            else: 
                                known_block = self.block_detections[block_idx]
                                known_block['confidence'] += 1
                                known_block['size_num'] += 1
                                known_block['frames_since_last_detection'] = 0

                                confidence_weight = (1 - (1/known_block['confidence']))
                                corrective_weight = 1 - confidence_weight

                                size_confidence_weight = (1 - (1/known_block['size_num']))
                                size_corrective_weight = 1 - size_confidence_weight
                                known_block['centroid'] = (confidence_weight * known_block['centroid'] + corrective_weight * block_info['centroid']).astype(int)
                                known_block['box'] = block_info['box']

                                # Update orientation of the block (dx, dy, theta) over time 
                                # to track changes. Reject sharp Movements between frames 
                                if known_block['prev_orientation'] is not None: 
                                    orientation_diff = known_block['orientation'][2] - block_info['orientation'][2]
                                    if orientation_diff < np.deg2rad(150): 
                                        known_block['prev_orientation'] = known_block['orientation']
                                        known_block['orientation'] = block_info['orientation']
                                        

                                # known_block['size_num'] =int(size_confidence_weight * known_block['size_num'] + size_corrective_weight * block_info['size_num'])
                                known_block['size_num'] = block_info['size_num']
                    else:

                        # Draw error contour
                        cv2.drawContours(
                            raw_image_to_show, [approx], 0, (255, 0, 0), 3)
                        # M = cv2.moments(approx)

                        # if M["m00"] != 0:
                        #     cx = int(M["m10"] / M["m00"])
                        #     cy = int(M["m01"] / M["m00"])
                        #     cv2.putText(raw_image, (str(len(
                        #         approx))), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                        num_undefined += 1

            # print(f"# Filtered ({num_valid}), P({num_parallelograms}) \t C({num_circles}) \t U({num_undefined})")

            # Then draw all boxes and centroids
            # print("SHOW ALL BLOCKS") 
            for block_idx, block in enumerate(self.block_detections): 

                # Collect box data 
                box = block["box"] 
                color = block["color"]
                color_int = block['color_int']

                # Draw bounding box 
                cv2.drawContours(raw_image_to_show, [box], 0, (0, 255, 255), 2)  # Yellow
                cv2.putText(raw_image_to_show, (str(color) + '_' + str(block['size_num'])+ '_' +  str(block['tfa'])+ '_ori' + str(np.round(np.rad2deg(block['orientation'][2]),2))), (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_int, 2)

                # Draw orientation 
                x_vec = (block["centroid"], block["centroid"] + block["orientation"][0])
                y_vec = (block["centroid"], block["centroid"] + block["orientation"][1])

                x_vec_p0 = tuple(x_vec[0])
                x_vec_p1 = tuple(x_vec[1])
                y_vec_p0 = tuple(y_vec[0])
                y_vec_p1 = tuple(y_vec[1])

                cv2.line(raw_image_to_show, x_vec_p0, x_vec_p1, (255, 0, 0), 5)
                cv2.line(raw_image_to_show, y_vec_p0, y_vec_p1, (0, 255, 0),5 )

                # Draw centroid 
                centroid = block["centroid"] 
                cv2.circle(raw_image_to_show, (centroid[0], centroid[1]),
                    1, (102, 255, 0), 2)


            self.BlockFrame = raw_image_to_show

        # self.BlockFrame = raw_image

    # Checks to see if a block is already detected (in the previous list of detections)
    #     block_info = {
    #     'label': num_valid,
    #     'color': str(color),
    #     'centroid': prime_points[4, :],
    #     'world_points': world_points_array[4, :],
    #     'box': prime_points[:4, :],
    #     'size': size_type,    # big or small
    #     'detected_in_last_frame': 1, 
    #     'frames_since_last_detection': 0

    # }
    def is_already_detected(self, block_info): 
        
        # Each block has a centroid 
        centroid = block_info["centroid"]
        color = block_info["color"]

        # Each block in self.block_detections also has a centroid 
        # Have some threshold for distance, called self.same_block_pixel_thresh
        for block_index, block_candidate in enumerate(self.block_detections): 
            if block_candidate["color"] == color: 
                if np.linalg.norm(block_candidate["centroid"] - centroid) < self.same_block_pixel_thresh: 
                    return block_index
        
        return -1
    

    def block_lookup(self, mouse_uv): 

        # Each block has a centroid, we're trying to see if there's one 
        # close to mouse_uv 

        for block_index, block_candidate in enumerate(self.block_detections): 
            if np.linalg.norm(block_candidate['centroid'] - mouse_uv) < self.same_block_pixel_thresh: 
                return True, block_candidate
            
        return False, None
    

    def clean_up_block_detections(self): 

        stale_frame_threshold = 10

        for i, block in enumerate(self.block_detections): 
            block["frames_since_last_detection"] += 1

            if block["frames_since_last_detection"] > stale_frame_threshold: 
                self.block_detections.pop(i)


    def is_rectangle(self, contour, area, min_aspect_ratio=0.9, max_aspect_ratio=1.1):
        """
        Check if contour is a rectangle.
        """

        is_parallelogram = False
        is_square = False
        is_rectangle = False

        cnt = contour
        # INPUT: cnt, epsilon, bool for closed shape
        approx = cv2.approxPolyDP(cnt, 0.07*cv2.arcLength(cnt, True), True)

        if len(approx) == 4:
            is_parallelogram = True
            # self.Circle = False
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= min_aspect_ratio and ratio <= max_aspect_ratio:
                is_square = True
                is_rectangle = False
            else:
                is_square = False
                is_rectangle = True

        # elif ((len(approx) > 8) and (area > 30)):
        #     self.Circle = True
        #     self.Parallelogram = False

        return is_parallelogram, is_square, is_rectangle, approx

        # peri = cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # if len(approx) == 4:  # Rectangles will have 4 vertices after approximation
        #     (x, y, w, h) = cv2.boundingRect(approx)
        #     aspectRatio = w / float(h)
        #     if min_aspect_ratio <= aspectRatio <= max_aspect_ratio:
        #         return True
        # return False

    def is_circular(self, contour, threshold=0.822):
        """
        Check if contour is a rectangle.
        """
        is_circle = False
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)

        circularity = 4*np.pi*area / (peri**2)

        # print(f'circularity: {circularity}')

        # if circularity >= threshold:
        #     is_circle = True

        if abs(circle_area - area) < 500:
            is_circle = True

        return is_circle
    
    def calculate_block_orientation(self, box, centroid): 

        # Box is a 4x2 array of U,V points corresponding to vertices 
        # centroid is a 1x2 array of U,V points corresponding to the centroid location 

        # Calculate the distances between all vertices 
        distances = [] 
        for i in range(4):
            for j in range(i+1, 4): 
                d = np.linalg.norm(box[i, :] - box[j, :])
                distances.append((d, i, j)) # Back vertices with to understand structure

        # Then sort the distances
        # First two largest distance will be the diagonal 
        # Largest distance will be first 
        distances.sort(reverse=True, key=lambda x: x[0])

        # Then we get width and height 
        width = distances[2][0]
        height = distances[4][0]

        # Get the indices of the width and height 
        width_indices = (distances[2][1], distances[2][2]) # Nodes 1 and 2 making up width
        height_indices = (distances[4][1], distances[4][2]) # Nodes 1 and 2 making up height 

        # Then calculate unit directions of both blocks 
        dx = box[width_indices[1]] - box[width_indices[0]]
        dy = box[height_indices[1]] - box[height_indices[0]]

        # Calculate theta (in the ground frame) 
        orientation = np.arctan2(dx[1], dx[0])

        return dx, dy, orientation

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        raw_depth = self.DepthFrameRGB.copy()

        # gray = cv2.cvtColor(raw_depth, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(raw_depth, (5, 5), 0)
        # canny = cv2.Canny(blur, 100,200)

        self.BlockFrame = blur

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        uncorrected_image = self.VideoFrame.copy()
        uncorrected_depth = self.DepthFrameRGB.copy()

        if self.camera_calibration_loaded == False:
            self.GridFrame = uncorrected_image
            return

        # Reshape grid points to be single rows
        # So there are n rows for n grid points
        # each row has 2 columns, which is ground truth
        board_points = self.grid_points.T.reshape(-1, 2)

        # To get 3D Points, append 3rd column of zeros to the whole thing
        board_points_3D = np.column_stack(
            (board_points, np.zeros(board_points.shape[0])))

        # To get homogenoues points, append a 4th column of ones on top of that
        board_points_homogenous = np.column_stack(
            (board_points_3D, np.ones(board_points.shape[0])))

        # Transpose to get favorable orientation
        # now columns are num points, rows are elements of homogenous coordinates
        board_points_world_homogenous = board_points_homogenous.T
        board_points_camera_homogenous = self.extrinsic_matrix @ board_points_world_homogenous
        board_points_camera_z_vals = board_points_camera_homogenous[2, :]

        board_points_uv = np.zeros((2, board_points_world_homogenous.shape[1]))

        for i in range(board_points_world_homogenous.shape[1]):

            # UV coordinate (homogenous, 1 is 3rd element)
            point_uv_homogeneous = (1/board_points_camera_z_vals[i]) * (
                self.intrinsic_matrix @ board_points_camera_homogenous[:3, i])
            board_points_uv[:, i] = point_uv_homogeneous[:2]

        # Apply the homography
        new_img = cv2.warpPerspective(
            uncorrected_image, self.homography, (uncorrected_image.shape[1], uncorrected_image.shape[0]))

       # Apply homography of depth frame as well
        depth_frame_homography = cv2.warpPerspective(
            uncorrected_depth, self.homography, (uncorrected_depth.shape[1], uncorrected_depth.shape[0]))

        # Convert the circle coordinates to UV in frame post homography
        board_points_uv_opencv_format = np.array(
            [board_points_uv.T], dtype=np.float32)
        board_points_uv_projected_opencv_format = cv2.perspectiveTransform(
            board_points_uv_opencv_format, self.homography)
        board_points_uv_projected_normal_format = board_points_uv_projected_opencv_format[0]

        # Save additional convenient frames
        # Grid frame no points is projected (homography) without grid points
        # DepthFrame homography is depth frame projected (homography)
        self.GridFrameNoPoints = deepcopy(new_img)
        self.DepthFrameHomography = deepcopy(depth_frame_homography)

        for i in range(board_points_uv_projected_normal_format.shape[0]):
            point_float = board_points_uv_projected_normal_format[i, :]
            point_int = (int(point_float[0]), int(point_float[1]))

            cv2.circle(new_img, point_int, 5,
                       (0, 0, 255), 1)

        self.GridFrame = new_img

    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()

        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        YELLOW = (255, 255, 0)

        for detection in msg.detections:
            point = (int(detection.centre.x), int(detection.centre.y))
            cv2.circle(modified_image, point, 3, GREEN, -1)

            # draw edges of the tags
            corners = detection.corners
            for i in range(4):
                cv2.line(modified_image,
                         (int(corners[i].x), int(corners[i].y)),
                         (int(corners[(i+1) % 4].x),
                          int(corners[(i+1) % 4].y)),
                         BLUE,
                         3)

            # draw the ID of the tags
            id = str(detection.id)
            cv2.putText(modified_image, "ID:"+id, (int(corners[2].x), int(corners[2].y) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)

            if self.apriltag_poses is not None:
                if detection.id in self.apriltag_poses["tag_ids"]:
                    # print(f'using {detection.id}id')
                    tag_data = self.apriltag_poses[detection.id]["uvd_poses"]

                    # Red is corner 1
                    # Blue is corner 2
                    # Green is Corner 3
                    # Yellow is corner 4

                    for i in range(4):
                        if i == 0:
                            color = RED
                        elif i == 1:
                            color = BLUE
                        elif i == 2:
                            color = GREEN
                        elif i == 3:
                            color = YELLOW

                        point_u = int(tag_data[i + 1, 0])
                        point_v = int(tag_data[i + 1, 1]) + 10
                        point = (point_u, point_v)

                        cv2.circle(modified_image, point, 6, color, -1)
                        cv2.putText(modified_image, str(i+1), point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)

        self.TagImageFrame = modified_image


class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.distortion_coefficients = np.array(data.d)
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Block window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                self.camera.blockDetector()
                # self.camera.detectBlocksInDepthImage()
                grid_frame = self.camera.convertQtGridFrame()
                block_frame = self.camera.convertQtBlockFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame, block_frame)
                # comment this out when run this file alone.
                self.executor.spin_once()
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                               cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Block window",
                               cv2.cvtColor(self.camera.BlockFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass

        self.executor.shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
