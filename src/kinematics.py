"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import modern_robotics as mr
import pybullet as p


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def clamp90deg(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi/2:
        angle -= np.pi
    while angle <= -np.pi/2:
        angle += np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """

    # Define basic transformation (identity)
    T = np.eye(4)

    # First link to compute
    link_index = 1

    # print("JOINT ANGLES")
    # print(joint_angles)

    # Compute transformations up to link
    while link_index <= link:
        current_dh_params = deepcopy(dh_params[link_index-1])

        # If this is the wrist rotation joint, the setpoint angle
        # must be canceled out in the negative direction
        # (see diagram)
        # print(f'pre change: {current_dh_params[3]}, link={link_index}')
        # if link_index == 4:
        # current_dh_params[3] -= joint_angles[link_index - 1]
        # else:
        current_dh_params[3] += joint_angles[link_index - 1]

        # print(f'post change: {current_dh_params[3]}, link={link_index}')
        # Clamp alplha and theta
        current_dh_params[1] = clamp(current_dh_params[1])
        current_dh_params[3] = clamp(current_dh_params[3])

        # print(f'post clamp: {current_dh_params[3]}, link={link_index}')
        # Calculate intermediate transformation (for link # link_index)
        intermediate_transformation = get_transform_from_dh(*current_dh_params)

        # Post multiply this transformation onto the larger scale transformation
        T = T @ intermediate_transformation

        # Update link index
        link_index += 1

    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """

    return np.array([
        [np.cos(theta), -1 * np.sin(theta) * np.cos(alpha),
         np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -1 *
         np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])


def get_euler_angles_from_T(T, order='zyz'):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    # Register transformation rotation as scipy rotation object
    rotation = R.from_matrix(T[:3, :3])

    # Return in euler order specified
    return rotation.as_euler(order)


def get_pose_from_T(T, order='zyz'):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """

    euler_angles = get_euler_angles_from_T(T, order)
    position = T[:3, 3]

    six_dof_pose = [position[0], position[1], position[2],
                    euler_angles[0], euler_angles[1], euler_angles[2]]

    return six_dof_pose


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_planar(dh_params, pose):

    # Desired position (x, y, z)
    desired_position = pose[:3]

    # print(f"Desired position: {desired_position}")

    # Desired rotation (theta, phi, psi) in zyz
    desired_rotation = pose[3:]

    # Check!! Proper range
    q1 = np.arctan2(desired_position[1], desired_position[0]) - np.pi/2

    # print("q1 before clamp:", np.rad2deg(q1))

    q1 = clamp(q1)

    # Assume
    q5 = 0

    # Fake arm length
    l1 = 205.73 / 1000

    # Shoulder length
    l2 = 200 / 1000

    # Wrist to EE length
    l3 = (66 + 65 + (43.15 * 4/5)) / 1000

    clamped_q3 = False

    # Angle between fake arm and real arm
    psi = np.arctan(50/200)

    # Get planar coordinates
    x_desired = np.sqrt(
        (desired_position[0] ** 2) + (desired_position[1] ** 2)) / 1000
    y_desired = (desired_position[2] / 1000) - (103.91/1000)

    # print(f"x_d {x_desired}")
    # print(f"y_d {y_desired}")
    # print(f'desired rotation theta = {desired_rotation[0]}')


    reachable, theta1_corrected, theta2_corrected, theta3_corrected, x_c, y_c = IK_planar_recompute(x_desired, y_desired, desired_rotation[0], l1, l2, l3, elbow_up=True)
    if not reachable or np.abs(theta2_corrected) > np.pi/2 or np.abs(theta3_corrected) > np.pi/2:
        reachable, theta1_corrected, theta2_corrected, theta3_corrected, x_c, y_c = IK_planar_recompute(x_desired, y_desired, desired_rotation[0], l1, l2, l3, elbow_up=False)

    # print(f'Desired pose: {[pose]} is {reachable} reachable.')

    q2 = theta1_corrected
    q3 = theta2_corrected
    q4 = theta3_corrected
    q5 = desired_rotation[1] # Screw axis of gripper 

    return np.array([[q1, q2, q3, q4, q5]])



def IK_planar_recompute(x_d, y_d, desired_rotation, l1, l2, l3, elbow_up=True): 

    # Goal points 
    x_c = x_d - l3*np.cos(-1 * desired_rotation)
    y_c = y_d + l3*np.sin(-1 * desired_rotation)

    
    # Detect if point unreachable 
    if np.sqrt(np.power(x_c, 2) + np.power(y_c, 2)) > (l1 + l2):
        return False, None, None, None, None, None
    elif np.sqrt(np.power(x_c, 2) + np.power(y_c, 2)) < np.abs(l1 - l2):
        return False, None, None, None, None, None
    
    #  Two options for theta 2 given elbow up elbow down position
    theta2_simple_1 = np.arccos((np.power(x_c, 2) + np.power(y_c, 2) - np.power(l1, 2) - np.power(l2, 2))/(2*l1*l2))
    theta2_simple_2 = -1 * np.arccos((np.power(x_c, 2) + np.power(y_c, 2) - np.power(l1, 2) - np.power(l2, 2))/(2*l1*l2))
    
    # print(f"theta 2 1: {np.rad2deg(theta2_simple_1)}")
    # print(f"theta 2 2: {np.rad2deg(theta2_simple_2)}")
    
    # Pick which one to use
    if elbow_up: 
        theta2_simple = theta2_simple_1
    else: 
        theta2_simple = theta2_simple_2
    
    theta1_simple = np.arctan(y_c/x_c) - np.arctan((l2*np.sin((theta2_simple)))/(l1 + l2*np.cos(theta2_simple)))
    theta3_simple = (desired_rotation - (theta1_simple + theta2_simple)) 
    
    # Finally, do correction for armlab arm
    psi = np.arctan(50/200)
    
    # Correction for theta 1
    theta1_corrected = np.pi/2 - psi - theta1_simple 
    
    # Correction for theta 2 
    theta2_corrected = -1*(theta2_simple + (np.pi/2 - psi))
    
    # Correction for theta 3
    theta3_corrected = -1 * theta3_simple
    
    # Finally, check once more if there are any invalid angles
    # if np.abs(theta2_corrected) > np.deg2rad(135): 
    #     return False, None, None, None, None, None
    # elif np.abs(theta3_corrected) > np.pi/2: 
    #     return False, None, None, None, None, None
    
    return True, theta1_corrected, theta2_corrected, theta3_corrected, x_c, y_c

