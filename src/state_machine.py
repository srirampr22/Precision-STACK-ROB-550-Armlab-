"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from rclpy.logging import get_logger
import json
import keyboard
import pickle
import sys
import os
from copy import deepcopy
from kinematics import IK_planar
import pandas as pd

# custom imports
import cv2


class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [[[-np.pi/2,                -0.5,         -0.3,              0.0,                0.0], "open"],
                          [[0.75*-np.pi/2,        0.5,          0.3,     -
                              np.pi/3,         np.pi/2], "open"],
                          [[0.5*-np.pi/2,         -0.5,         -0.3,
                              np.pi / 2,                 0.0], "open"],
                          [[0.25*-np.pi/2,        0.5,          0.3,     -
                            np.pi/3,          np.pi/2], "open"],
                          [[0.0,                          0.0,          0.0,
                            0.0,                 0.0], "open"],
                          [[0.25*np.pi/2,        -0.5,         -0.3,
                            0.0,         np.pi/2], "open"],
                          [[0.5*np.pi/2,           0.5,           0.3,     -
                            np.pi/3,                0.0], "open"],
                          [[0.75*np.pi/2,       -0.5,          -0.3,
                            0.0,          np.pi/2], "open"],
                          [[np.pi/2,                 0.5,
                            0.3,           -np.pi/3,          0.0], "open"],
                          [[0.0,                         0.0,           0.0,            0.0,               0.0], "open"]]

        self.current_waypoint_num = 0
        self.moving_to_waypoint = False
        self.arm_moving_speed_rad_s = 0.5
        self.current_error = 0
        self.current_error_angles = np.zeros((5,))
        self.gripper_state_last = "open"

        # self.waypoints = [
        #                 [-1.56926239,  0.37122336, -0.10124274,  0.82067972,  0.00613592]]

        self.recorded_pos = []
        self.flag = False

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "record":
            self.enter_recording()

        if self.next_state == "stop_recording":
            self.stop_recording()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "testing":
            self.testing_state()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "pick_and_place":
            self.pick_and_place()

        if self.next_state == "pick":
            self.pick()

        if self.next_state == "place":
            self.place()

        if self.next_state == 'sort_task':
            self.sort_task()
        
        if self.next_state == 'calibrate_z_sag': 
            self.calibrate_z_sag()

        if self.next_state == "line_task":
            self.line_task()

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

        # H_desired = np.array([[0, 1,  0,     -5],
        #               [1, 0,  0,      4],
        #               [0, 0, -1, 1.6858],
        #               [0, 0,  0,      1]])
        # print(f'trying IK: H_desired={H_desired}')

        # theta_list, result = set_ee_pose_matrix(H_desired)

        # print(f'result: {result}; theta_list={theta_list}')

    def testing_state(self):

        # Dedicated state to test functions
        self.status_message = "State: TESTING - Waiting for completion"
        self.current_state = "testing"
        self.next_state = "idle"

        # goal_poses = [[-39.42631593, 340.54512933, 997.07764333, -np.pi/2, 0, 0]]
                    

        # joint_angles = IK_geometric(self.rxarm.dh_params, goal_pose)

        # print(joint_angles)

        # self.rxarm.arm.set_joint_positions(
        #         joint_angles[0]
        # )

        
        # for pose in goal_poses:
            # pose[2] -= 37
        # joint_angles = IK_planar(self.rxarm.dh_params, pose)
        joint_angles = [[103.54, 26.37, 37.71, -65.83, 0.0],
                        [-120.54, 26.37, 37.71, -65.83, 0.0]]

        # joint_angles = np.deg2rad(joint_angles)
        # print(f'joint_nagles from IK: {joint_angles}')
        # print(
        #     f" Radians: {joint_angles}, Degrees: {np.rad2deg(joint_angles)}")
        # rate = 200
        for joint in joint_angles:
            # print(joint_angles[0] - i)
            joint = np.deg2rad(joint)
            print(joint)
            time.sleep(1)
            self.rxarm.arm.set_joint_positions(joint, moving_time=4)
            time.sleep(1)

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def enter_recording(self):

        if self.current_state != "record":
            self.status_message = "State: Record - Manually guide the rxarm and hit record"
            self.current_state = "record"
            self.rxarm.disable_torque()

        self.next_state = "record"

    def pick_and_place(self):

        self.status_message = "PICK AND PLACE MODE - Click a cube and have the arm pick it up. Click again to have it be put down"
        self.camera.new_click = False
        self.current_state = "pick_and_place"
        self.next_state = "pick"

    def pick(self):
        self.current_state = "pick"
        self.status_message = "PICK MODE - Click a cube and have the arm pick it up. Click again to have it be put down"

        # print(f"Click: {self.camera.new_click}")

        if self.camera.new_click:

            # If there's been a click, get the location of click
            pick_position_uv = self.camera.last_click
            pick_position = self.camera.last_click_world

            # Figure out if there's a block at that position
            is_block, block_info = self.camera.block_lookup(pick_position_uv) 

            # If the block is detected, calculate arm screw
            arm_screw = 0

            if is_block: 
                arm_screw = block_info['orientation'][2]


            # Add angles to the pick position to represent orientation
            pick_position.append(-1*np.pi/2)
            pick_position.append(arm_screw)
            pick_position.append(0)

            above_position_joint_angles = np.array([0, 0, -1*np.pi/4, 0, 0])


            # Navigate to above position
            time.sleep(1)
            # joint_angles[0][1] -= np.deg2rad(15)
            self.rxarm.arm.set_joint_positions(above_position_joint_angles)
            time.sleep(1)

            # Navigate to grab position
            time.sleep(1)
            # pick_position[2] += 20
            # joint_angles = IK_planar(self.rxarm.dh_params, pick_position)
            # self.rxarm.arm.set_joint_positions(joint_angles[0])
            self.command_arm_pose(pick_position)
            time.sleep(1)
            self.rxarm.gripper.grasp()
            time.sleep(1)

            # Navigate to above position
            time.sleep(1)
            # joint_angles[0][1] -= np.deg2rad(15)
            self.rxarm.arm.set_joint_positions(above_position_joint_angles)
            time.sleep(1)

            self.camera.new_click = False
            self.next_state = "place"

        else:
            self.next_state = "pick"

    def pick_comp(self, pick_position, sleep_time=1, move_time=1.0, grab_angle=-np.pi/2, tilt_angle=0):

        pick_position = list(pick_position)

        # If there's been a click, get the location of click
        pick_position.append(grab_angle)
        pick_position.append(0)
        pick_position.append(0)

        above_position = deepcopy(pick_position)
        above_position[2] += 100

        # print(f"ABOVE POSITION: {above_position}")
        # print(f"PICK POSITION: {pick_position}")

        # Navigate to above position
        time.sleep(sleep_time)
        joint_angles = IK_planar(self.rxarm.dh_params, above_position)

        print(f'pick angles: {joint_angles}')

        if (joint_angles == None).any():
            print(f'uncheable, going 0 angle')
            pick_position[3] = -np.pi/4
            above_position[3] = -np.pi/4
            above_position[2] += 30
            joint_angles = IK_planar(self.rxarm.dh_params, above_position)

            if (joint_angles == None).any(): 
                return False
            
        else: 
            print("Assigning tilt angles.") 
            print(f"Heading is {np.rad2deg(joint_angles[0][0])}") 
            print(f"Pre-wrist angle: {joint_angles[0][4]}")
            joint_angles[0][4] = tilt_angle - joint_angles[0][0]
            print(f"Post-wrist angles {joint_angles[0][4]} for requested tilt of {np.rad2deg(tilt_angle)}")

        # joint_angles[0][1] -= np.deg2rad(15)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)

        # Navigate to grab position
        time.sleep(sleep_time)
        pick_position[2] += 20
        joint_angles = IK_planar(self.rxarm.dh_params, pick_position)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)
        self.rxarm.gripper.grasp()
        time.sleep(sleep_time)

        # Navigate to above position
        time.sleep(sleep_time)
        joint_angles = IK_planar(self.rxarm.dh_params, above_position)
        joint_angles[0][1] -= np.deg2rad(15)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)
        return True

    def place_comp(self, place_position, sleep_time=1, move_time=1.0, grab_angle=-np.pi/2, align_x=False):

        place_position = list(place_position)

        # If there's been a click, get the location of click
        print("PPPPPPPPPPPlace position:",place_position)

        place_position.append(grab_angle)
        place_position.append(0)
        place_position.append(0)

        above_position = deepcopy(place_position)
        # above_position[2] += 50

        # Navigate to above position
        time.sleep(sleep_time)
        joint_angles = IK_planar(self.rxarm.dh_params, above_position)

        if (joint_angles == None).any():
            print(f'uncheable, going 0 angle')
            place_position[3] = -np.pi/4
            # above_position[3] = 0
            # above_position[2] += 30
            joint_angles = IK_planar(self.rxarm.dh_params, above_position)

            if (joint_angles == None).any(): 
                return False
            
        if align_x: 
            joint_angles[0][4] = -2*joint_angles[0][0]
            
        print(f'place angles: {joint_angles}')

        joint_angles[0][1] -= np.deg2rad(15)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)

        # Navigate to grab position
        time.sleep(sleep_time)
        joint_angles = IK_planar(self.rxarm.dh_params, place_position)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)
        self.rxarm.gripper.release()
        time.sleep(sleep_time)

        # Navigate to above position
        time.sleep(sleep_time)
        joint_angles = IK_planar(self.rxarm.dh_params, above_position)
        joint_angles[0][1] -= np.deg2rad(30)
        self.rxarm.arm.set_joint_positions(joint_angles[0], moving_time=move_time)
        time.sleep(sleep_time)
        return True

    def place(self):
        self.current_state = "place"
        self.status_message = "PLACE MODE - Click a cube and have the arm pick it up. Click again to have it be put down"

        if self.camera.new_click:

            # If there's been a click, get the location of click
            place_position_uv = self.camera.last_click
            place_position = self.camera.last_click_world

            # Add angles to the place position 
            place_position.append(-1*np.pi/2) 
            
            # Calculate screw axis angle using some trig
            arm_screw = np.arctan2(place_position[1], place_position[0])
            place_position.append(arm_screw)
            place_position.append(0)

            # Set default above position of gripper
            above_position_joint_angles = np.array([0, 0, -1*np.pi/4, 0, 0])

            # Navigate to above position
            time.sleep(1)
            self.rxarm.arm.set_joint_positions(above_position_joint_angles)
            time.sleep(1)

            # Navigate to grab position
            time.sleep(1)
            # joint_angles = IK_planar(self.rxarm.dh_params, place_position)
            # self.rxarm.arm.set_joint_positions(joint_angles[0])
            self.command_arm_pose(place_position)
            time.sleep(1)
            self.rxarm.gripper.release()
            time.sleep(1)

            # Navigate to above position
            time.sleep(1)            # joint_angles[0][1] -= np.deg2rad(15)
            self.rxarm.arm.set_joint_positions(above_position_joint_angles)
            time.sleep(1)

            self.camera.new_click = False
            self.next_state = "idle"

        else:
            self.next_state = "place"

    def blocks_to_sort(self):
        detected_blocks = self.camera.block_detections
        # Getting all the blocks in the scene
        
        # num_sorted_blocks = {"big": 0, 'small': 0}
    
        small_blocks = []
        big_blocks = []
        
        for block in detected_blocks:

            if block['size_num'] == 1:
                if block['world_points'][1] > 0:
                    big_blocks.append(block)
                # else: 
                    # num_sorted_blocks['big'] += 1

            elif block['size_num'] == 0:
                if block['world_points'][1] > 0:
                    small_blocks.append(block)
                # else:
                    # num_sorted_blocks['small'] += 1
            # else:
            #     return -1
        print(f'small blocks: {small_blocks}')
        print(f'big blocks: {big_blocks}')

        return small_blocks, big_blocks

    def sort_task(self):
        self.status_message = "State: Sort Task - Performing 1st Task of competition"
        self.current_state = "sort_tast"
        self.next_state = 'idle'

        num_sorted_blocks = {"big": 0, 'small': 0}
        move_time = 1.2

        place_height = 50

        small_blocks, big_blocks = self.blocks_to_sort()

        sleep_time = 0.2

        # self.testing_state()
        time.sleep(1)

        self.rxarm.initialize()
        time.sleep(2)

        while small_blocks != [] or big_blocks != []:

            # Do bigs first
            if big_blocks != []:
                sorting_block = big_blocks[0]
                place_position = [
                    150 + 75*(num_sorted_blocks['big'] % 3), -90, place_height]

                if num_sorted_blocks['big'] >= 4:
                    place_position[1] += 50

            # All bigs done, do small
            else:
                sorting_block = small_blocks[0]
                place_position = [-150 - 75 *(num_sorted_blocks['small'] % 3), -90, place_height]

                if num_sorted_blocks['small'] >= 4:
                    place_position[1] += 50

            pick_position = sorting_block['world_points']
            pick_position[2] -= 20

            print(f'pick_position: {pick_position}')
            print(f'place_position: {place_position}')

            valid = self.pick_comp(pick_position, sleep_time=sleep_time, move_time=move_time)

            if not valid: 
                continue

            time.sleep(sleep_time)
           
            valid = self.place_comp(place_position, sleep_time=sleep_time, move_time=move_time)

            if not valid: 
                continue

            time.sleep(sleep_time)

            self.rxarm.initialize()

            time.sleep(sleep_time)

            if big_blocks != []:
                num_sorted_blocks["big"] += 1
            else:
                num_sorted_blocks["small"] += 1

            small_blocks, big_blocks = self.blocks_to_sort()

            print(f'num_sorted_blocks: {num_sorted_blocks}')


    def get_linedUp_blocks(self):
        detected_blocks = self.camera.block_detections
        # Getting all the blocks in the scene
        
        # num_sorted_blocks = {"big": 0, 'small': 0}
        # size of small block is 27.6 
        # size of large block is 38
        
    
        small_blocks = [None] * 6
        big_blocks = [None] * 6
        
        for block in detected_blocks:

            if block['color'] == 'yellow':
                color_index = 2
            elif block['color'] == 'green':
                color_index = 3
            elif block['color'] == 'orange':
                color_index = 1
            elif block['color'] == 'red':
                color_index = 0
            elif block['color'] == 'blue':
                color_index = 4
            elif block['color'] == 'purple':
                color_index = 5

            if not self.block_is_in_line_position(block, color_index): 
                if block['size_num'] == 1:
                    big_blocks[color_index] = block

                elif block['size_num'] == 0:
                    small_blocks[color_index] = block
                # else:
                    # num_sorted_blocks['small'] += 1
            # else:
            #     return -1
        print(f'small blocks: {small_blocks}')
        print(f'big blocks: {big_blocks}')

        return small_blocks, big_blocks
    
    def get_assigned_line_position(self, block, index): 
        big_line_y = 300 
        big_line_x_start = 200
        big_line_x_gap = 60
        
        small_line_y = 175
        small_line_x_start = 200
        small_line_x_gap = 60

        # Get position of block 
        position = block['world_points']

        # Check which position the block should be in line
        size_num = block['size_num'] 

        # Big block 
        if size_num == 1: 
            assigned_position_x = big_line_x_start - (big_line_x_gap)*index
            assigned_position_y = big_line_y
        # Small block 
        elif size_num == 0: 
            assigned_position_x = small_line_x_start - (small_line_x_gap)*index
            assigned_position_y = small_line_y


        return np.array([assigned_position_x, assigned_position_y])
    
    def block_is_in_line_position(self, block, index): 

        # Get position of block 
        position = block['world_points']
        assigned_position = self.get_assigned_line_position(block, index)

        distance_threshold = 15 
        distance = np.linalg.norm(assigned_position - position[:2]) 
        
        if distance <= distance_threshold: 
            return True
        else: 
            return False


    def line_task(self):
        self.status_message = "State: Line Task - Performing 3rd Task of competition"
        self.current_state = "line_task"
        self.next_state = 'idle'

        num_sorted_blocks = {"big": 0, 'small': 0}
        move_time = 1.2
        place_height = 50
        small_blocks, big_blocks = self.get_linedUp_blocks()
        sleep_time = 0.2


        self.rxarm.initialize()
        time.sleep(2)

        while small_blocks != ([None] * 6) or big_blocks != ([None] * 6):

            # Do bigs first
            if big_blocks != ([None] * 6):
                for block_idx, block in enumerate(big_blocks): 
                    if block is not None: 

                        # If block in this color is not none, 
                        # Get it's place position 
                        pick_position = block['world_points']
                        place_position = np.append(self.get_assigned_line_position(block, block_idx), place_height)
                        break
                

            # All bigs done, do small
            else:
                for block_idx, block in enumerate(small_blocks): 
                    if block is not None: 

                        # If block in this color is not none, 
                        # Get it's place position 
                        pick_position = block['world_points']
                        place_position = np.append(self.get_assigned_line_position(block, block_idx), place_height)
                        break
                
            # Lower Z for correction
            pick_position[2] -= 20

            print(f'pick_position: {pick_position}')
            print(f'place_position: {place_position}')

            valid = self.pick_comp(pick_position, sleep_time=sleep_time, move_time=move_time, tilt_angle=block['orientation'][2])

            if not valid: 
                continue

            time.sleep(sleep_time)
           
            valid = self.place_comp(place_position, sleep_time=sleep_time, move_time=move_time, align_x=True)

            if not valid: 
                continue

            time.sleep(sleep_time)

            self.rxarm.initialize()

            time.sleep(sleep_time)

            # if big_blocks != []:
            #     num_sorted_blocks["big"] += 1
            # else:
            #     num_sorted_blocks["small"] += 1

            small_blocks, big_blocks = self.get_linedUp_blocks()

            # print(f'num_sorted_blocks: {num_sorted_blocks}')
        

    def record_waypoints(self):

        # self.status_message = "State: Idle - Waiting for input"
        # self.current_state = "idle"

        # if not self.flag:
        #     positions = self.rxarm.get_positions()
        #     gripper_state = self.rxarm.gripper_state
        #     self.recorded_pos.append(positions)
        #     print("POS!!",self.recorded_pos)
        #     self.flag = True

        self.status_message = "State: Record - Manually guide the rxarm and hit record"
        self.current_state = "record"

        self.rxarm.disable_torque()

    def record_current_waypoint(self, gripper=''):
        current_pos = self.rxarm.get_positions()

        if gripper == '':
            raise Exception("WAYPOINT DOES NOT HAVE GRIPPER STATE")
            # Record waypoint with gripper state

        # if not self.flag:
        with open("waypoints.json", "a") as f:
            json.dump((current_pos.tolist(), gripper), f)
            f.write('\n')

    def stop_recording(self):

        if self.current_state != "stop_recording":
            self.current_state = "stop_recording"
            self.status_message = "State: Recording Stopped"
            self.rxarm.enable_torque()
            # current_pos = self.rxarm.get_positions()
            # with open("waypoints.json", "a") as f:
            #     json.dump(current_pos.tolist(), f)
            #     f.write('\n')

        self.next_state = "stop_recording"

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.current_state = "execute"
        self.status_message = "State: Execute - Executing motion plan"

        # Get current waypoint and arm position
        current_waypoint = self.waypoints[self.current_waypoint_num][0]
        current_waypoint_gripperstate = self.waypoints[self.current_waypoint_num][1]
        current_pose = self.rxarm.get_positions()

        # Current norm difference between waypoint being moved to and current arm position
        self.current_error_angles = current_waypoint - current_pose
        self.current_error = np.linalg.norm(self.current_error_angles)
        error_thresh = 0.1  # Error norm threshold

        # Calculate biggest joint error
        self.biggest_joint_error = np.amax(
            np.absolute(self.current_error_angles))

        # Calculate arm move time if needed
        arm_move_time = (self.biggest_joint_error) / \
            self.arm_moving_speed_rad_s
        arm_accel_time = 1/3 * arm_move_time

        # if arm_move_time > 3:
        #     arm_move_time = 3
        # if arm_accel_time > 0.5:
        #     arm_accel_time = 0.5

        # arm_move_time = 2
        # arm_accel_time = 0.5

        if self.moving_to_waypoint == False:
            print(f"Current Index (For Error): {self.current_waypoint_num}")
            print(f"Error used: {self.biggest_joint_error}")
            print(f"New Arm Move Time: {arm_move_time}")
            print(f"New Arm Accel Time: {arm_accel_time}")
            self.rxarm.arm.set_joint_positions(
                current_waypoint,
                arm_move_time,
                arm_accel_time,
                blocking=False
            )
            # self.rxarm.set_positions(current_waypoint)

            self.moving_to_waypoint = True
            self.next_state = "execute"
        else:

            # If arm has reached waypoint it was moving towards
            if self.current_error < error_thresh:

                # Check to see if we need to change gripper state
                gripper_state = current_waypoint_gripperstate  # either "open" or "closed"

                if gripper_state == "open" and self.gripper_state_last == "closed":
                    self.rxarm.gripper.release()
                    self.gripper_state_last = "open"
                elif gripper_state == "closed" and self.gripper_state_last == "open":
                    self.rxarm.gripper.grasp()
                    self.gripper_state_last = "closed"

                # Set next waypoint
                self.current_waypoint_num += 1

                # If waypoint is end of list, end execute
                if self.current_waypoint_num == len(self.waypoints):
                    self.moving_to_waypoint = False
                    self.current_waypoint_num = 0  # Allow for multiple runs
                    self.next_state = "idle"

                # Otherwise, move to next waypoint
                else:
                    self.moving_to_waypoint = False
                    self.next_state = "execute"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Looking for April Tag detections"

        # Parent Apriltag Detection Object
        detections = self.camera.tag_detections

        # Identify U, V, D coordinates of apriltag points
        # Hardcode in ground truth for apriltag points
        # Solve PnP and get extrinsic (make sure this is a function we can replace )
        # Update self.camera.extrinsic_matrix

        # apriltag_poses dict that will be passed
        # to any of the solve PnP functions
        # ASSUME point order is CENTER, CORNER1, CORNER2, CORNER3, CORNER4
        apriltag_poses = {
            1: {
                "ground_truth": np.array([[-250, -25, 0],  # CENTRE
                                          [-237.5, -12.5, 0],  # CORNER 1
                                          [-237.5, -37.5, 0],  # CORNER 2
                                          [-262.5, -37.5, 0],  # CORNER 3
                                          [-262.5, -12.5, 0]])  # CORNER 4
            },
            2: {
                "ground_truth": np.array([[250, -25, 0],  # CENTRE
                                          [262.5, -12.5, 0],  # CORNER 1
                                          [262.5, -37.5, 0],  # CORNER 2
                                          [237.5, -37.5, 0],  # CORNER 3
                                          [237.5, -12.5, 0]])  # CORNER 4
            },
            3: {
                "ground_truth": np.array([[250, 275, 0],  # CENTRE
                                          [262.5, 287.5, 0],  # CORNER 1
                                          [262.5, 262.5, 0],  # CORNER 2
                                          [237.5, 262.5, 0],  # CORNER 3
                                          [237.5, 287.5, 0]])  # CORNER 4
            },
            4: {
                "ground_truth": np.array([[-250, 275, 0],  # CENTRE
                                          [-237.5, 287.5, 0],  # CORNER 1
                                          [-237.5, 262.5, 0],  # CORNER 2
                                          [-262.5, 262.5, 0],  # CORNER 3
                                          [-262.5, 287.5, 0]])  # CORNER 4
            }
        }

        # # Add additional tags referenced to central 4
        apriltag_poses[5] = {}
        apriltag_poses[5]["ground_truth"] = deepcopy(
            apriltag_poses[4]["ground_truth"])
        apriltag_poses[5]["ground_truth"][:,
                                          0] = apriltag_poses[5]["ground_truth"][:, 0] - 125
        apriltag_poses[5]["ground_truth"][:,
                                          1] = apriltag_poses[5]["ground_truth"][:, 1] + 25
        apriltag_poses[5]["ground_truth"][:, 2] = 157  # height above 0

        apriltag_poses[6] = {}
        apriltag_poses[6]["ground_truth"] = deepcopy(
            apriltag_poses[3]["ground_truth"])
        apriltag_poses[6]["ground_truth"][:,
                                          0] = apriltag_poses[6]["ground_truth"][:, 0] + 75
        apriltag_poses[6]["ground_truth"][:,
                                          1] = apriltag_poses[6]["ground_truth"][:, 1] - 125
        apriltag_poses[6]["ground_truth"][:, 2] = 65  # height above 0

        apriltag_poses[7] = {}
        apriltag_poses[7]["ground_truth"] = deepcopy(
            apriltag_poses[1]["ground_truth"])
        apriltag_poses[7]["ground_truth"][:,
                                          0] = apriltag_poses[7]["ground_truth"][:, 0] - 125
        apriltag_poses[7]["ground_truth"][:,
                                          1] = apriltag_poses[7]["ground_truth"][:, 1] - 75
        apriltag_poses[7]["ground_truth"][:, 2] = 242.8  # height above 0

        apriltag_poses["tag_ids"] = []

        for i, detection in enumerate(detections.detections):
            # Note down which IDs have been seen
            apriltag_poses["tag_ids"].append(detection.id)

        num_tags_used = len(apriltag_poses["tag_ids"])

        # Construct change of basis to determine locations of apriltag corners uniquely.
        # We need at least tags 1 2 and 4 in the detection scene
        centroids = np.zeros([2, num_tags_used])

        # Get all centroids in matrix
        for i, detection in enumerate(detections.detections):

            # Only consider first 4 tags
            if detection.id not in apriltag_poses["tag_ids"]:
                print(f'{detection.id} NOT SEEN')
                continue

            centroid_u = int(detection.centre.x)
            centroid_v = int(detection.centre.y)
            centroids[:, detection.id -
                      1] = np.array([centroid_u, centroid_v]).T

        # Construct change of basis matrix
        apriltag_frame = np.zeros([2, 2])

        # Horizontal vector
        v_horizontal = (centroids[:, 1] - centroids[:, 0])
        v_horizontal /= np.linalg.norm(v_horizontal)

        # Vertical
        v_vertical = (centroids[:, 3] - centroids[:, 0])
        v_vertical /= np.linalg.norm(v_vertical)

        # Horizontal vector
        apriltag_frame[:, 0] = v_horizontal

        # Vertical vector
        apriltag_frame[:, 1] = v_vertical

        for i, detection in enumerate(detections.detections):

            # Only use apriltags 1-4 for calibration
            if detection.id not in apriltag_poses["tag_ids"]:
                continue

            # Create blank array of apriltag poses in camera frame
            camera_poses = np.zeros((5, 3))
            uvd_poses = np.zeros((5, 3))

            # Get camera intrinsic matrix
            K = self.camera.intrinsic_matrix
            K_inv = np.linalg.inv(K)

            # First do centroid
            centroid_u = int(detection.centre.x)
            centroid_v = int(detection.centre.y)
            centroid_depth = self.camera.DepthFrameRaw[centroid_v][centroid_u]

            # Calculate centroid in apriltag frame for corner determination
            centroid_u_tag_frame, centroid_v_tag_frame = apriltag_frame @ np.array(
                [centroid_u, centroid_v]).T

            centroid_uvd_pose = np.array(
                [centroid_u, centroid_v, centroid_depth], dtype=np.float32)
            centroid_pose = (centroid_depth *
                             (K_inv @ np.array([centroid_u, centroid_v, 1]).T))

            # Update camera pose matrices
            uvd_poses[0, :] = centroid_uvd_pose
            camera_poses[0, :] = centroid_pose

            # Go through all corners and calculate poses
            for corner in detection.corners:
                corner_u = int(corner.x)
                corner_v = int(corner.y)
                corner_depth = self.camera.DepthFrameRaw[corner_v][corner_u]

                # Figure out what the corner ID is (assumed that top right in
                # world frame is #1, move clockwise)
                # This is done in the apriltag frame in order to universally determine
                # which corner is which
                corner_u_tag_frame, corner_v_tag_frame = apriltag_frame @ np.array(
                    [corner_u, corner_v]).T

                corner_id = -1
                if (corner_u_tag_frame > centroid_u_tag_frame) and (corner_v_tag_frame > centroid_v_tag_frame):
                    corner_id = 1
                elif (corner_u_tag_frame > centroid_u_tag_frame) and (corner_v_tag_frame < centroid_v_tag_frame):
                    corner_id = 2
                elif (corner_u_tag_frame < centroid_u_tag_frame) and (corner_v_tag_frame < centroid_v_tag_frame):
                    corner_id = 3
                elif (corner_u_tag_frame < centroid_u_tag_frame) and (corner_v_tag_frame > centroid_v_tag_frame):
                    corner_id = 4
                else:
                    raise Exception(
                        f"Apriltag with ID {detection.id} has an unidentifiable corner.")

                corner_uvd_pose = np.array(
                    [corner_u, corner_v, corner_depth], dtype=np.float32)
                corner_pose = (
                    corner_depth * (K_inv @ np.array([corner_u, corner_v, 1]).T))

                # Fill camera pose matrices
                uvd_poses[corner_id, :] = corner_uvd_pose
                camera_poses[corner_id, :] = corner_pose

            # After centroid and corners all have pose, update dict
            apriltag_poses[detection.id]["uvd_poses"] = uvd_poses
            apriltag_poses[detection.id]["camera_poses"] = camera_poses

        self.camera.apriltag_poses = apriltag_poses
        print("apriltagposes:", self.camera.apriltag_poses)
        # extrinsic, _ = solve_pnp(apriltag_poses)
        # A_affine = self.recover_homogeneous_transform_pnp_ippe(apriltag_poses)
        A_affine = self.recover_homogeneous_transform_pnp(apriltag_poses)

        self.camera.extrinsic_matrix = A_affine

        print("CAMERA EXTRINSIHC:", A_affine)

        # Hard coded ground truth (world frame)
        border_corners_ground_truth = np.array(
            [[500, 475, 0, 1], [500, -175, 0, 1], [-500, -175, 0, 1], [-500, 475, 0, 1]]).T

        # Figure out where these points lie in the first camera frame
        border_corners_camera_poses = A_affine @ border_corners_ground_truth
        border_corners_camera_z_vals = border_corners_camera_poses[2, :]

        # Construct matrix of UV coordinates for border corners
        border_corners_uv = np.zeros((2, border_corners_ground_truth.shape[1]))

        for i in range(border_corners_ground_truth.shape[1]):
            # UV coordinate (homogenous, 1 is 3rd element)
            point_uv_homogeneous = (1/border_corners_camera_z_vals[i]) * (
                self.camera.intrinsic_matrix @ border_corners_camera_poses[:3, i])
            border_corners_uv[:, i] = point_uv_homogeneous[:2]

        # z_c = border_corners_camera_poses(2,:)

        # border_corners_uv1 = (1/z_c) * self.camera.intrinsic_matrix @ border_corners_camera_poses

        # Select source points to apply the homography transform from
        src_pts = border_corners_uv.T

        # Select destination points to apply the homography transform to

        aspect_ratio = 16/9
        px_correction_width = 200

        dest_pts = np.array([1280, 0,
                             1280, 720,
                             0, 720,
                             0, 0,]).reshape((4, 2))

        if px_correction_width != 0:

            # Top right point
            dest_pts[0, 0] -= px_correction_width/2
            dest_pts[0, 1] += ((px_correction_width/2) / aspect_ratio)

            # Bottom right point
            dest_pts[1, 0] -= px_correction_width/2
            dest_pts[1, 1] -= ((px_correction_width/2) / aspect_ratio)

            # Bottom left
            dest_pts[2, 0] += px_correction_width/2
            dest_pts[2, 1] -= ((px_correction_width/2) / aspect_ratio)

            # Top left
            dest_pts[3, 0] += px_correction_width/2
            dest_pts[3, 1] += ((px_correction_width/2) / aspect_ratio)

            dest_pts = dest_pts.astype(int)

        self.camera.homography = cv2.findHomography(src_pts, dest_pts)[0]

        camera_calibration_dict = {
            "intrinsic_matrix": self.camera.intrinsic_matrix,
            "extrinsic_matrix": self.camera.extrinsic_matrix,
            "distortion_coefficients": self.camera.distortion_coefficients,
            "homography": self.camera.homography
        }

        with open('../config/camera_calibration.pickle', 'wb') as handle:
            pickle.dump(camera_calibration_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        self.camera.camera_calibration_loaded = True
        self.status_message = "Calibapriltag_posesration - Completed Calibration"

    def recover_homogeneous_affine_opencv(self, apriltag_poses):

        tags_used = len(apriltag_poses["tag_ids"])

        # Camera points (stack for data input)
        camera_poses_points = np.array(
            [apriltag_poses[i]['camera_poses'] for i in apriltag_poses['tag_ids']])
        camera_poses_points = np.reshape(camera_poses_points, (5*tags_used, 3))

        # World points (stack for data input)
        world_poses_points = np.array(
            [apriltag_poses[i]['ground_truth'] for i in apriltag_poses['tag_ids']])
        world_poses_points = np.reshape(world_poses_points, (5*tags_used, 3))

        _, T, _ = cv2.estimateAffine3D(
            camera_poses_points, world_poses_points, confidence=0.99)
        # print(T)
        return np.row_stack((T, (0.0, 0.0, 0.0, 1.0)))

    def recover_homogeneous_transform_pnp(self, apriltag_poses):

        tags_used = len(apriltag_poses["tag_ids"])

        # Camera points (stack for data input)
        image_poses_points = np.array(
            [apriltag_poses[i]['uvd_poses'] for i in apriltag_poses['tag_ids']])
        image_poses_points = np.reshape(image_poses_points, (5*tags_used, 3))
        image_poses_points = np.delete(image_poses_points, -1, axis=1)

        print("imageposepoints:", apriltag_poses["tag_ids"])

        # World points (stack for data input)
        world_poses_points = np.array(
            [apriltag_poses[i]['ground_truth'] for i in apriltag_poses['tag_ids']])
        world_poses_points = np.reshape(world_poses_points, (5*tags_used, 3))

        print("worldpoints:", world_poses_points.shape)

        [_, R_exp, t] = cv2.solvePnP(
            world_poses_points,
            image_poses_points,
            self.camera.intrinsic_matrix,
            self.camera.distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        R, _ = cv2.Rodrigues(R_exp)

        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

    def recover_homogeneous_transform_pnp_ippe(self, apriltag_poses):

        tags_used = len(apriltag_poses["tag_ids"])

        # Camera points (stack for data input)
        image_poses_points = np.array(
            [apriltag_poses[i]['uvd_poses'] for i in apriltag_poses['tag_ids']])
        image_poses_points = np.reshape(image_poses_points, (5*tags_used, 3))
        image_poses_points = np.delete(image_poses_points, -1, axis=1)

        # World points (stack for data input)
        world_poses_points = np.array(
            [apriltag_poses[i]['ground_truth'] for i in apriltag_poses['tag_ids']])
        world_poses_points = np.reshape(world_poses_points, (5*tags_used, 3))

        [_, R_exp, t] = cv2.solvePnP(
            world_poses_points,
            image_poses_points,
            self.camera.intrinsic_matrix,
            self.camera.distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE
        )

        R, _ = cv2.Rodrigues(R_exp)

        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))
    
    
    def calibrate_z_sag(self): 

        # Update state variables 
        self.next_state = "idle"
        self.current_state = "calibrate_z_sag"
        self.status_message = "Z Sag Calibration. Sampling cylindrical points."

        # Generate a number of positions to sample throughout the space
        datapoints = pd.DataFrame(columns=['r', 'x_d', 'y_d', 'z_d', 'x', 'y', 'z'])

        # Import these params to generate grid
        # Fake arm length
        l1 = 205.73

        # Shoulder length
        l2 = 200

        # Elbow to ee length
        l3 = (66 + 65 + (43.15 * 4/5))

        # max_r
        # This defines our sweep of x/y
        max_r = l1 + l2 + l3
        r_multiplier = 0.8

        theta = np.linspace(0, np.pi, num=5) 
        z_sample = np.linspace(40, 300, num=5)

        while r_multiplier >= 0.4: 
            for z_desired in z_sample: 
                for theta_i in theta: 
                    x_desired = max_r * r_multiplier * np.cos(theta_i) 
                    y_desired = max_r * r_multiplier * np.sin(theta_i)

                    desired_pose = [x_desired, y_desired, z_desired, 0, 0, 0]

                    print(f" desired pose: {desired_pose}")

                    # Then calculate angles for this position 
                    joint_angles = IK_planar(self.rxarm.dh_params, desired_pose)

                    print(f"Joint angles: {joint_angles}")

                    state_pre = np.array(self.rxarm.get_ee_pose())

                    # Navigate to that position 
                    self.rxarm.arm.set_joint_positions(joint_angles[0]) 

                    # Wait and sleep 
                    time.sleep(1)

                    # Collect true pose of system 
                    state_true = np.array(self.rxarm.get_ee_pose())

                    # Check to see if the arm couldn't get to this location, throw out datapoint
                    if np.linalg.norm(state_true - state_pre) < 0.1: 
                        continue

                    # Add to datapoints 
                    datapoints.loc[len(datapoints.index)] = {
                        'r': max_r * r_multiplier,
                        'x_d': x_desired, 
                        'y_d': y_desired, 
                        'z_d': z_desired, 
                        'x': state_true[0], 
                        'y': state_true[1], 
                        'z': state_true[2]
                    }

                # After a round, reduce multiplier 
            r_multiplier -= 0.2

        # After all this is done, save to csv 
        datapoints.to_csv('../calibration_backups/end_effector.csv')

    def command_arm_pose(self, pose): 

        # print desired pose
        

        # Perform correction if possible
        # if self.rxarm.sag_lookup_table is not None: 

        #     # Find closest request for lookup 
        #     # These are the two values we search on
        #     r = np.sqrt(np.power(pose[0], 2) + np.power(pose[1], 2))
        #     z = pose[2]

        #     # Get unique values in lookup 
        #     r_unique = (self.rxarm.sag_lookup_table['r'].unique())
        #     z_unique = (self.rxarm.sag_lookup_table['zd'].unique())

        #     if z >= np.min(z_unique) and r >= np.min(r_unique): 
        #         print("A lookup table was found and values are in range") 

        #         # Get value that's closest to ours
        #         mindex_z = np.argmin(z_unique-z) 
        #         mindex_r = np.argmin(r_unique-r) 

        #         minval_z = z_unique[mindex_z]
        #         minval_r = r_unique[mindex_r]

        #         # Then request this correction from the original table 
        #         z_correction = self.rxarm.sag_lookup_table[
        #             (self.rxarm.sag_lookup_table['r'] == minval_r) & 
        #             (self.rxarm.sag_lookup_table['zd'] == minval_z)
        #         ]['zerror'].values[0]

        #         # Check if the applied correction is nan, if it is, use simple correction 
        #         if np.isnan(z_correction): 
        #             print("The requested correction was NaN (unmodeled region), so performing simple correction instead")
        #             z_correction = 10
        #         else: 
        #             print(f"The applied correction in z will be: {z_correction}")
                
        #         # Then apply correction 
        #         pose[2] -= z_correction

        #     elif z < np.min(z_unique) and r > 200:
        #         print("A lookup table was found, but the values are out of range") 
        #         print("However, in range for simple z correction") 
        #         pose[2] -= 10 

        #     else: 
        #         print("A lookup table was found but the values are out of range") 
        # else: 
        #     print("A lookup table was not found")
        # if self.rxarm.sag_lines is not None: 

        #     # Find closest request for lookup 
        #     # These are the two values we search on
        #     r = np.sqrt(np.power(pose[0], 2) + np.power(pose[1], 2))
        #     z = pose[2]

        #     z_correction = -1
        #     for line in self.rxarm.sag_lines[-1:]: 

        #         # If we are at the minimum threshold 
        #         if line[0] < z and line[3] < r: 

        #             # Perform correction 
        #             z_correction = ((r/1000)*line[1] + line[2])(1000) # mm -> m -> mm
        #             print(f"Line found to perform correction. Correction is {z_correction}")
                    
        #             break
        #     if z_correction == -1: 
        #         print("No lines matched requested pose, only doing a simple correction") 
        #         z_correction = 60

        #     # Apply additional, shitty side correction 
        #     if pose[0] < 0: 
        #         z_correction += 20

        #     pose[2] -= z_correction 
        # else: 
        #     print("No line table was found")

        theta_case = 0
        r_case = 0

        r = np.sqrt(np.power(pose[0], 2) + np.power(pose[1], 2))
        z = pose[2]
        theta = np.arctan2(pose[1], pose[0]) - np.pi/2

        # Define correction value 
        z_correction = 0

        # Segment on regions of the grid 
        if (theta > 0 and theta < np.pi/3) or (theta < -np.pi/3): 
            print("theta case 1") 
            theta_case = 1
            # z_correction = 5
        elif (theta > np.pi/3): 
            print("theta case 2") 
            theta_case = 2
            # z_correction = 10
        elif (theta < 0 and theta > -np.pi/3): 
            print("theta case 3") 
            theta_case = 3
            # z_correction = 5

        if (r > 300): 
            print("r case 1") 
            r_case = 1
            # z_correction += 15
        elif (r > 200): 
            print("r case 2") 
            r_case = 2
            # z_correction += 2
        elif (r > 100): 
            print("r case 3") 
            r_case = 3
            # z_correction += 0


        if (theta_case == 1 & r_case == 1): 
            z_correction = 15
        elif (theta_case == 2 & r_case == 1): 
            z_correction = 15
        elif (theta_case == 3 & r_case == 1): 
            z_correction = 10
        elif (theta_case == 1 & r_case == 2): 
            z_correction = 15
        elif (theta_case == 2 & r_case == 2): 
            z_correction = 10
        elif (theta_case == 3 & r_case == 2): 
            z_correction = 5
        elif (theta_case == 1 & r_case == 3): 
            z_correction = 5
        elif (theta_case == 2 & r_case == 3): 
            z_correction = 2
        elif (theta_case == 3 & r_case == 3): 
            z_correction = 0

        # Print desired pose and metrics 
        print(f"Desired pose: {pose}") 
        print(f"r = {r}") 
        print(f"z = {z}") 
        print(f"theta = {theta}") 
        print(f"Correction: {z_correction}")

        # Then add the correction 
        pose[2] += z_correction

        # Then request the movement 
        joint_angles = IK_planar(self.rxarm.dh_params, pose) 

        # Check to see if this position is unreachable 
        if np.any(joint_angles[0] == None): 
            # Try and find the other configuration before giving up
            print("Original pose unreachable, trying other wrists") 
            if pose[3] == -1*np.pi/2: 
                pose[3] = -1*np.pi/4
                pose[4] = 0

                joint_angles = IK_planar(self.rxarm.dh_params, pose)
                if np.any(joint_angles[0] == None): 
                    pose[3] = 0
                    pose[4] = 0
                    joint_angles = IK_planar(self.rxarm.dh_params, pose)

            elif pose[3] == 0: 
                pose[3] = -1*np.pi/2

            joint_angles = IK_planar(self.rxarm.dh_params, pose) 

            # If this still doesn't work, throw an error
            if np.any(joint_angles[0] == None):
                print("Pose is unreachable from both wrist postures.") 

        print(f"Final angles: {joint_angles[0]}")

        pre_angles = deepcopy(self.rxarm.get_positions()) 
        pre_angles[0] = joint_angles[0][0]
        angles = joint_angles[0]

        self.rxarm.arm.set_joint_positions(pre_angles) 
        time.sleep(1)
        self.rxarm.arm.set_joint_positions(angles)

        return joint_angles[0]



    """ TODO """

    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    currentStateMessage = pyqtSignal(str)
    currentWaypointIndex = pyqtSignal(int)
    currentTargetError = pyqtSignal(float)
    currentTargetErrorAngles = pyqtSignal(list)

    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm = state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            self.currentStateMessage.emit(self.sm.current_state)
            self.currentTargetError.emit(self.sm.current_error)
            self.currentTargetErrorAngles.emit(
                self.sm.current_error_angles.tolist())
            self.currentWaypointIndex.emit(self.sm.current_waypoint_num)
            time.sleep(0.05)
