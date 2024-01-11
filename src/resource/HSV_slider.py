#!/usr/bin/env python3

import numpy as np
import cv2
import pyrealsense2 as realsense

def nothing(x):
    pass

class RealSenseCamera:
    def __init__(self):
        self.pipeline = realsense.pipeline()
        self.config = realsense.config()
        self.config.enable_stream(realsense.stream.color, 1280, 720, realsense.format.bgr8, 30)
        self.pipeline.start(self.config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    # Setup RealSense camera
    camera = RealSenseCamera()
    # Create window for HSV sliders
    cv2.namedWindow('HSV Color Palette with RealSense L515 Feed')
    cv2.createTrackbar('H_l', 'HSV Color Palette with RealSense L515 Feed', 0, 179, nothing)
    cv2.createTrackbar('S_l', 'HSV Color Palette with RealSense L515 Feed', 0, 255, nothing)
    cv2.createTrackbar('V_l', 'HSV Color Palette with RealSense L515 Feed', 0, 255, nothing)

    cv2.createTrackbar('H_u', 'HSV Color Palette with RealSense L515 Feed', 0, 179, nothing)
    cv2.createTrackbar('S_u', 'HSV Color Palette with RealSense L515 Feed', 0, 255, nothing)
    cv2.createTrackbar('V_u', 'HSV Color Palette with RealSense L515 Feed', 0, 255, nothing)


    try:
        while True:
            color_image = camera.get_frame()
            if color_image is not None:
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

                # Get slider values
                h_l = cv2.getTrackbarPos('H_l', 'HSV Color Palette with RealSense L515 Feed')
                s_l = cv2.getTrackbarPos('S_l', 'HSV Color Palette with RealSense L515 Feed')
                v_l = cv2.getTrackbarPos('V_l', 'HSV Color Palette with RealSense L515 Feed')

                h_u = cv2.getTrackbarPos('H_u', 'HSV Color Palette with RealSense L515 Feed')
                s_u = cv2.getTrackbarPos('S_u', 'HSV Color Palette with RealSense L515 Feed')
                v_u = cv2.getTrackbarPos('V_u', 'HSV Color Palette with RealSense L515 Feed')


                lower_bound = np.array([h_l, s_l, v_l])  # Adjust as needed
                upper_bound = np.array([h_u,s_u, v_u])  # Adjust as needed
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                result = cv2.bitwise_and(color_image, color_image, mask=mask)

                cv2.imshow('HSV Color Palette with RealSense L515 Feed', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()