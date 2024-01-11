#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
import json
from functools import partial

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread

# Manual imports 
from resource.data_monitor import data_monitor


""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None, dh_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(self.start_calibration)
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(self.gripper_open)
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(self.gripper_close)
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(self.execute_default_trajectory)
        
        # ADDED BUTTONS FOR CHECKPOINT 1'
        self.ui.btnUser5.setText("Start Recording")
        self.ui.btnUser5.clicked.connect(self.start_recording)
        self.ui.btnUser6.setText("Execute Teacher Trajectory")
        self.ui.btnUser6.clicked.connect(self.execute_teacher_trajectory)
        self.ui.btnUser7.setText("Execute Custom Test") 
        self.ui.btnUser7.clicked.connect(self.enter_testing_state)
        self.ui.btnUser8.setText("Enter Pick and Place") 
        self.ui.btnUser8.clicked.connect(self.enter_pick_and_place)
        self.ui.btnUser9.setText("Enter Sort Task")
        self.ui.btnUser9.clicked.connect(self.sort_task)
        self.ui.btnUser10.setText("Calibrate Z Sag") 
        self.ui.btnUser10.clicked.connect(self.calibrate_z_sag)
        self.ui.btnUser11.setText("Enter Line Task")
        self.ui.btnUser11.clicked.connect(self.line_task)
        # self.ui.btnUser6.clicked.connect(self.replay_trajectory)


        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.currentStateMessage.connect(
            self.currentStateMessage)
        self.StateMachineThread.currentTargetError.connect(
            self.currentTargetError)
        self.StateMachineThread.currentTargetErrorAngles.connect(
            self.currentTargetErrorAngles)
        self.StateMachineThread.currentWaypointIndex.connect(
            self.currentWaypointIndex
        )
        self.StateMachineThread.start()


        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()


        """
            Attached data monitor
        """
        # self.data_monitor = data_monitor()
        # self.data_monitor.exec_() 

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

        # print(msg)
        # if msg == '"State: Record - Manually guide the rxarm and hit record"':
        #     self.ui.btnUser5.setText("Stop Recording")
        #     self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'record'))


    @pyqtSlot(str) 
    def currentStateMessage(self, msg): 

        # Our current state
        if msg == "record": 
            
            if self.ui.btnUser5.text() == "Start Recording": 
                # Updates the text of the start recording button to "stop recording" 
                # when we enter record states
                self.ui.btnUser5.setText("Stop Recording")
                self.ui.btnUser5.clicked.disconnect(self.start_recording)
                self.ui.btnUser5.clicked.connect(self.stop_recording)

                # Updates the text of the execute trajectories button to "record waypoint" 
                # when we are in record states
                self.ui.btnUser6.setText("Clear Waypoints") 
                self.ui.btnUser6.clicked.disconnect(self.execute_teacher_trajectory)
                self.ui.btnUser6.clicked.connect(self.clear_waypoints)

                # Updates the text and functionality of open gripper /close gripper to map to waypoints 
                self.ui.btnUser2.setText("Record Open Waypoint") 
                self.ui.btnUser2.clicked.disconnect(self.gripper_open) 
                self.ui.btnUser2.clicked.connect(self.record_waypoint_open) 

                self.ui.btnUser3.setText("Record Closed Waypoint") 
                self.ui.btnUser3.clicked.disconnect(self.gripper_close) 
                self.ui.btnUser3.clicked.connect(self.record_waypoint_close)

        # if the current state is not record
        else: 

            # Updates the text of the buttons when going from record state back to idle
            if self.ui.btnUser5.text() == "Stop Recording": 

                # Updates the "start/stop" record button 
                self.ui.btnUser5.setText("Start Recording") 
                self.ui.btnUser5.clicked.disconnect(self.stop_recording) 
                self.ui.btnUser5.clicked.connect(self.start_recording) 

                # Updates the execute/record waypoint button 
                self.ui.btnUser6.setText("Execute Teacher Trajectory") 
                self.ui.btnUser6.clicked.disconnect(self.clear_waypoints) 
                self.ui.btnUser6.clicked.connect(self.execute_teacher_trajectory)

                # Updates the gripper open/close buttons 
                self.ui.btnUser2.setText("Open Gripper") 
                self.ui.btnUser2.clicked.disconnect(self.record_waypoint_open) 
                self.ui.btnUser2.clicked.connect(self.gripper_open) 

                self.ui.btnUser3.setText("Close Gripper") 
                self.ui.btnUser3.clicked.disconnect(self.record_waypoint_close)
                self.ui.btnUser3.clicked.connect(self.gripper_close) 

        # print(msg)

    @pyqtSlot(float) 
    def currentTargetError(self, msg): 
        # print(f"Current Error Norm: {msg}")
        
        data = {
            "error": msg
        }
        # self.data_monitor.update_readouts(data)

    @pyqtSlot(list) 
    def currentTargetErrorAngles(self, msg): 
        # print(f"Current Error Angles: {msg}")

        # data = {
        #     "error": np.amax(msg)
        # }
        # self.data_monitor.update_readouts(data)
        return 
    
    @pyqtSlot(int)
    def currentWaypointIndex(self, msg): 
        # print(f"Current waypoint index: {msg}")
        return 

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (1000 * pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (1000 * pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (1000 * pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image, block_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        
        num_checked = (int(self.ui.radioVideo.isChecked()) 
                + int(self.ui.radioDepth.isChecked())
                + int(self.ui.radioUsr1.isChecked())
                + int(self.ui.radioUsr2.isChecked())
                + int(self.ui.radioUsr3.isChecked()))

        # Assure that only one video feed is being pressed at once

        assert (num_checked == 1)


        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))
        if (self.ui.radioUsr3.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(block_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def start_calibration(self):
        self.ui.radioUsr1.click()
        self.sm.set_next_state('calibrate')

    def start_recording(self): 
        self.sm.set_next_state('record') 
    
    def stop_recording(self): 
        self.sm.set_next_state('stop_recording') 

    def execute_default_trajectory(self): 
        self.sm.set_next_state('execute')

    def enter_pick_and_place(self): 
        self.sm.set_next_state('pick_and_place') 

    def sort_task(self):
        self.sm.set_next_state('sort_task')

    def line_task(self):
        self.sm.set_next_state('line_task')

    def calibrate_z_sag(self): 
        self.sm.set_next_state('calibrate_z_sag') 

    def execute_teacher_trajectory(self):

        self.sm.waypoints = []

        with open("waypoints.json", "r") as f:
            for line in f:
                if line == "": 
                    continue
                else: 
                    position = json.loads(line.strip())
                    self.sm.waypoints.append(position)

        self.sm.set_next_state('execute')

    def enter_testing_state(self): 
        self.sm.set_next_state('testing')

    def record_waypoint_close(self):
        self.sm.record_current_waypoint(gripper='closed')
    
    def record_waypoint_open(self):
        self.sm.record_current_waypoint(gripper='open') 

    def clear_waypoints(self):
        with open("waypoints.json", "w") as f:
            f.close()


    def gripper_close(self): 
        self.rxarm.gripper.grasp() 
    
    def gripper_open(self): 
        self.rxarm.gripper.release() 

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        mouse_position = mouse_event.pos()

        # Get mouse position point 
        point = np.array([mouse_position.x(), mouse_position.y()])
        # print(f"POINT: {point}")

        # Check if we are in the grid frame, if so, need to unwarp points to regular camera UV
        if self.ui.radioUsr2.isChecked() or self.ui.radioUsr3.isChecked(): 
            
            # Put point in weird opencv format
            point_cv = np.array([point], dtype=np.float32) 
            point_cv = np.array([point_cv])

            # print(f"POINT CV UNTRANSFORMED: {point_cv}")

            # Need to do an unwarping of point 
            homography_inverse = np.linalg.inv(self.camera.homography) 
            
            point_cv = cv2.perspectiveTransform(point_cv, homography_inverse)

            # print(f"POINT CV TRANSFORMED: {point_cv}")
            point = point_cv[0][0].astype(int)

            # Check to make sure that the unwrapped point is valid. 
            point[0] = np.clip(point[0], 0, 1279).astype(int)
            point[1] = np.clip(point[1], 0, 719).astype(int) 

            if  self.camera.bool_detect == True:
                self.ui.DetectedBlock.setText("Found it")
                # print("FOUND ITTT")
            else:
                self.ui.DetectedBlock.setText("no bueno")
                # print("NOT FOUND")


            # print(f"UPDATED POINT: {point}")


        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.DepthFrameRaw[point[1]][point[0]]
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (point[0], point[1], z))

            # Camera Intrinsics                          
            # K = [880.7213735911496, 0.0, 663.7897782822254, 0.0, 876.478410444169, 372.13260871961063, 0.0, 0.0, 1.0]
            # K = np.array([[902.187744140625, 0.0, 662.3499755859375], [0.0, 902.391357421875, 372.2278747558594], [0.0, 0.0, 1.0]])
            K = self.camera.intrinsic_matrix
            K_inv = np.linalg.inv(K)
            
            H_inv = np.linalg.inv(self.camera.extrinsic_matrix)

            # # Pixel coordinates
            u = point[0]  # Replace pt.x() with the actual pixel coordinate
            v = point[1]  # Replace pt.y() with the actual pixel coordinate

            # # print(X)
            camera_coords = (z * (K_inv @ np.array([u,v,1]).T))

            world_coord = H_inv @ np.array([camera_coords[0], camera_coords[1], camera_coords[2], 1]).T

            X = world_coord[0]
            Y = world_coord[1]
            Z = world_coord[2]

            
            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                    (X, Y, Z))
            self.ui.mousePositionWorld = [X, Y, Z]
            
    
            

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.last_click_world = self.ui.mousePositionWorld
        self.camera.new_click = True
        # print("CLICKKKKK",self.camera.last_click)

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'])
    app_window.show()

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
