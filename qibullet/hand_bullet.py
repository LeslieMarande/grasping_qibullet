import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import pybullet as p
import numpy as np
import json
import cv2
import time
import tools
from qibullet import SimulationManager
from qibullet import PepperVirtual
from numpy.linalg import inv

def mainGripper(object_file, grasp_points_path_file, min_time, min_quality,\
saving=True):
    """
    The end-effector tries to grasp the fixed object using physic properties,
    each grasp point is evaluate and all great ones are stocked in a file with its quality.

    Parameters:
        object_file - the object's urdf file in the folder object_data
        grasp_points_path_file - The path to the JSON file containing the valid grasp points
        min_time - the minimum time the object has not to fall to concider
        the grasp great
        min_quality - the minimum quality required to consider
        a grasp point great
    """
    f = open(grasp_points_path_file,"r")
    file = json.loads(f.read())
    f.close()
    gripper_name = file["parameters"]["gripper"]
    if gripper_name == "RGripper":
        gripper_name = "rGripper"
    elif gripper_name == "LGripper":
        gripper_name = "lGripper"
    else:
        print("Error: No such gripper")
        return

    grasp_points = []
    all_grasp_points = pickGraspPoints(grasp_points_path_file, min_quality)
    for grasp in all_grasp_points:
        pos = grasp[0]
        quaternion = grasp[1]
        grasp_points.append([pos,quaternion])

    print("--- Process all Grasp ---")
    grasp_points_file = grasp_points_path_file.split("/")[-1].split('.')[0]
    grasping(object_file, gripper_name, grasp_points, min_time, min_quality,
    grasp_points_file, saving=saving)
    return

def oneGrasp(object, object_constraint, pepper_gripper, gripper_name,
grasp_point, min_time):
    """
    Grasp the object and raise it with one specific grasp point and evaluate it
    in a simulation already created

    Parameters:
        object - id of the object given by loadURDF
        object_constraint - the constraint created for the object
        pepper_gripper - the gripper used (rGripper or lGripper)
        grasp_point - the grasp point desired for the end-effector
        min_time - the minimum time in seconds the object has not to fall
        to concider the grasp valid

    Returns:
        quality - the quality of the grasping
        [new_pos, quaternion] - The real 6D pose reached
    """
    # print("Grasp test:", grasp_point)
    if gripper_name == "rGripper":
        hand = "RHand"
    elif gripper_name == "lGripper":
        hand = "LHand"
    else:
        print("Error: No such gripper")
        return
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(1)

    pos=grasp_point[0]
    quaternion=grasp_point[1]

    cid = p.createConstraint(pepper_gripper.robot_model, -1, -1, -1, \
    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.5])

    #reset position to avoid collision problems
    p.resetBasePositionAndOrientation(pepper_gripper.robot_model,\
    [0, 0, 0.5], [0,0,0,1])
    time.sleep(.01)

    #Open the hand
    pepper_gripper.setAngles(hand, 1, 0.8)
    time.sleep(.03)

    ###Approache of the end-effector towards the object###
    # print('Grasp tested:', grasp_point)
    interval = 0.1 # start position
    current_x_pos = pos[0] + interval * np.sign(pos[0])
    current_y_pos = pos[1] + interval * np.sign(pos[1])
    current_z_pos = pos[2] + interval * np.sign(pos[2])
    precision = 0.0001

    x_done = False
    y_done = False
    z_done = False
    while 1:
        if not (abs(current_x_pos - pos[0]) < precision):
            # print("Move on x")
            new_pos = moveOnOneAxe(0, cid, current_x_pos, [pos[0], \
            current_y_pos, current_z_pos+1], quaternion)
            current_x_pos = new_pos[0]
        else:
            x_done=True

        if not (abs(current_y_pos - pos[1]) < precision):
            # print("Move on y")
            new_pos = moveOnOneAxe(1, cid, current_y_pos, [current_x_pos, \
            pos[1], current_z_pos+1], quaternion)
            current_y_pos = new_pos[1]
        else:
            y_done=True

        if not (abs(current_z_pos - pos[2]) < precision):
            # print("Move on z")
            new_pos = moveOnOneAxe(2, cid, current_z_pos + 1, [current_x_pos, \
            current_y_pos, pos[2]], quaternion)
            current_z_pos = new_pos[2] - 1
        else :
            z_done=True

        if (x_done and y_done and z_done):
            #Close enough
            break

    time.sleep(.1)

    #Close the hand
    pepper_gripper.setAngles(hand, 0, 0.8)
    time.sleep(.3)

    ###Evaluation of the grasp###
    initial_distance_object_endeffector = tools.getDistance(\
    p.getBasePositionAndOrientation(object)[0],\
    p.getBasePositionAndOrientation(pepper_gripper.robot_model)[0])
    p.removeConstraint(object_constraint)
    time.sleep(.3)

    ###Raise the object###
    # print("Try to raise the object ")
    current_z_pos = new_pos[2]
    start = time.time()
    while time.time() - start < min_time:
        time.sleep(.01)
        p.setGravity(0, 0, -10)
        z_pivot = [new_pos[0], new_pos[1], current_z_pos]
        p.changeConstraint(cid, z_pivot,\
        jointChildFrameOrientation=quaternion, maxForce=10)
        current_z_pos = current_z_pos + 0.002
    time.sleep(.01)
    final_distance_object_endeffector = tools.getDistance(\
    p.getBasePositionAndOrientation(object)[0],\
    p.getBasePositionAndOrientation(pepper_gripper.robot_model)[0])

    #Evaluation of the grasp point
    quality=evaluteGrasp(initial_distance_object_endeffector,\
     final_distance_object_endeffector)

    p.removeConstraint(cid)
    new_pos[2] = new_pos[2] - 1

    return [quality, [new_pos, quaternion]]

def grasping(object_file, gripper_name, grasp_points, min_time, min_quality=0,
grasp_points_file = None, saving=False):
    """
    Run grasping and evalution on each grasp point in the file

    Parameters:
        object_file - the object's urdf file in the folder object_data
        gripper_name - the name of the end-effector
        grasp_point - the grasp point reaches by the end-effector
        min_time - the minimum time the object has not to fall to concider
        the grasp valid
        min_quality - the minimum quality required to consider
        grasp_points_file - The name of the saved file

    Returns:
        quality - the quality of the grasping concidering the time
    """
    if saving:
            great_grasps = {}
            great_grasps["parameters"] = {}
            great_grasps["parameters"]["gripper"] = gripper_name
            great_grasps["parameters"]["min_quality"] = min_quality
            great_grasps["parameters"]["object"] = object_file
            great_grasps["parameters"]["min_time"] = min_time
            great_grasps["grasps"] = {}

    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)

    z_value = 1
    object = p.loadURDF(
              "object_data/"+object_file,
              [0, 0, z_value],
              [0, 0, 0, 1],
              globalScaling=1.0)
    time.sleep(.02)

    # time.sleep(15)
    grasp_nb = 0
    actual_qualities = []
    for grasp in grasp_points:
        pepper_gripper = simulation_manager.spawnGripperPepper(gripper_name,\
        client, spawn_ground_plane=True)

        print("-- New grasp:", grasp_nb,'--')
        # print(grasp)
        quality = 0
        pivot = np.array([np.array([0,0,0]), np.array([0,0,0,0])])

        ### Need to repete the same grasp and average the quality
        ### because the physic is not perfect
        repetition = 3
        for _ in range(repetition):
            object_constraint = p.createConstraint(object, -1, -1, -1,\
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
            p.resetBasePositionAndOrientation(object,[0,0,1],[0,0,0,1])
            p.resetBasePositionAndOrientation(pepper_gripper.robot_model,[0,0,0.5],\
            [0,0,0,1])
            grasp_done = oneGrasp(object, object_constraint, pepper_gripper,\
            gripper_name, grasp, min_time)
            quality += grasp_done[0]
            pivot += grasp_done[1]
        quality /= repetition
        pivot = np.array(pivot) / repetition
        pivot = [list(pivot[0]), list(pivot[1])]

        ###Need to remove and create again the model because of physic problems
        p.removeBody(pepper_gripper.robot_model)

        print("The grasp's quality is", float(quality))
        # print("Real grasp point done :", pivot)
        grasp_nb += 1

        if saving:
            ###Save great grasp in a file###
            quality = float(quality)

            if quality >= float(min_quality):
                if quality not in actual_qualities:
                    actual_qualities.append(quality)
                    great_grasps["grasps"][quality] = []
                great_grasps["grasps"][quality].append(list(pivot))

    if saving:
        f = open("../qibullet/graspPoints_data/JSON/graspPoints_qibullet/"+
        "grasp_qibullet_"+grasp_points_file+".json","w")
        # f = open("../qibullet/graspPoints_data/JSON/graspPoints_qibullet/"+
        # "grasp_qibullet__"+grasp_points_file+".json","w")
        f.write(json.dumps(great_grasps))
        f.close()
        print("File saved:","../qibullet/graspPoints_data/JSON/graspPoints_qibullet/"+
        "grasp_qibullet_"+grasp_points_file+".json")
    print("All grasp done, saved done")
    time.sleep(15)
    return

def evaluteGrasp(initial_distance_object_endeffector, final_distance_object_endeffector):
    """
    Evaluation of the grasp concidering only if the object is still in the hand
    or not after rasing it

    Parameters:
    initial_distance_object_endeffector - the distance between the object and
    the end effector before raised it
    final_distance_object_endeffector - the distance between the object and
    the end effector after raised it3

    Returns:
    quality - the quality depends on the position of the object
    """
    precision = 0.005
    if final_distance_object_endeffector - precision > initial_distance_object_endeffector:
        # print("Grasp failed")
        quality = 0
    else:
        # print("Grasp succed")
        quality = 1
    return quality

def pickGraspPoints(path_file, min_quality):
    """
    Pick the position and the quaternion of all great grasp points
    in a file concidering the quality

    Parameters:
        path_file - The path to the file which contain all grasp points
        min_quality - The minimum quality required to consider
        a grasp point valid

    Returns:
        grasp_points - All great grasp points in the file
        concidering the quality
    """
    grasp_points = []
    f = open(path_file,"r")
    file = json.loads(f.read())

    all_grasp_points = file['grasps']

    for quality in all_grasp_points:
        if float(quality) >= min_quality:
            for grasp in all_grasp_points[quality]:
                grasp_points.append(grasp)
    print("-------",len(grasp_points),"grasp points !")
    return grasp_points

def moveOnOneAxe(axe, cid, current_pos, pos, quaternion):
    """
    Change the pivot of the constraint considering one axe

    Parameters:
        axe - The id of considerin axe: 0 for x, 1 for y and 2 for z
        cid - The constraint
        current_pos - The current value of the choosen axe
        pos - The position of the base (list of 3 elements)
        quaternion - The actual quaternion

    Returns:
        pivot - The 3 new coordinates
    """
    time.sleep(.01)
    p.setGravity(0, 0, -10)
    step = 0.005 #step to go farword the object
    if axe == 0:
        current_pos = current_pos - step * np.sign(pos[0])
        pivot = [current_pos, pos[1], pos[2]]
    elif axe == 1:
        current_pos = current_pos - step * np.sign(pos[1])
        pivot = [pos[0], current_pos, pos[2]]
    elif axe == 2:
        current_pos = current_pos - step * np.sign(pos[2])
        pivot = [pos[0], pos[1], current_pos]
    else:
        print("Error: Not good number for axe, only 0, 1 or 2")

    p.changeConstraint(cid, pivot,\
    jointChildFrameOrientation=quaternion, maxForce=30)
    time.sleep(.01)
    return pivot

#Not finished
# def selectGraspPoints(grasp_points_path_file, min_quality, x_interval,
#  y_interval, z_interval, roll_interval, pitch_interval, yaw_interval):
#     """
#     Select some grasp points considering their 6D position
#
#     Parameters:
#         grasp_points_path_file - The path to the JSON file containing the valid
#         grasp points
#         min_quality - the minimum quality required to consider a grasp point
#         great
#         x_interval - a list [minimum_x, maximum_x], the x value of the grasp
#         has to be in this interval
#         y_interval - a list [minimum_y, maximum_y], the y value of the grasp
#         has to be in this interval
#         z_interval - a list [minimum_z, maximum_z], the z value of the grasp
#         has to be in this interval
#         roll_interval - a list [minimum_roll, maximum_roll], the roll value of
#         the grasp has to be in this interval
#         pitch_interval - a list [minimum_pitch, maximum_pitch], the pitch value
#          of the grasp has to be in this interval
#         yaw_interval - a list [minimum_yaw, maximum_yaw], the yaw value of
#         the grasp has to be in this interval
#
#     Returns:
#         valid_grasp_points - Grasp points with a 6D position
#         between all intervals
#     """
#     f = open(grasp_points_path_file,"r")
#     file = json.loads(f.read())
#     f.close()
#     grasp_points = pickGraspPoints(grasp_points_path_file, min_quality)
#
#     x_min = x_interval[0]
#     x_max = x_interval[1]
#     y_min = y_interval[0]
#     y_max = y_interval[1]
#     z_min = z_interval[0]
#     z_max = z_interval[1]
#
#     roll_min = roll_interval[0]
#     roll_max = roll_interval[1]
#
#     pitch_min = pitch_interval[0]
#     pitch_max = pitch_interval[1]
#
#     yaw_min = yaw_interval[0]
#     yaw_max = yaw_interval[1]
#
#     valid_grasp_points = []
#
#     for grasp in grasp_points:
#         pos = grasp[0]
#         if not (pos[0] >= x_min and pos[0] <= x_max):
#             continue
#         elif not (pos[1] >= y_min and pos[1] <= y_max):
#             continue
#         elif not (pos[2] >= z_min and pos[2] <= z_max):
#             continue
#
#         orientation = grasp[1]
#
#         angle_orientation = quaternion_to_euler(orientation[0], orientation[1],
#         orientation[2], orientation[3])
#         if not (angle_orientation[0] >= roll_min and pos[0] <= roll_max):
#             continue
#         elif not (angle_orientation[1] >= pitch_min and pos[1] <= pitch_max):
#             continue
#         elif not (angle_orientation[2] >= yaw_min and pos[2] <= yaw_max):
#             continue
#         else:
#             valid_grasp_points.append(grasp)
#
#     return valid_grasp_points

def quaternion_to_euler(x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z


start_all = time.time()

object_file = "cube_grasping.urdf"
grasp_points_path_file_simox = "../qibullet/graspPoints_data/JSON/graspPoints_"\
+"simoxCgal/1318grasp_points_RGripper.json"
# grasp_points_path_file_qibullet = "../qibullet/graspPoints_data/JSON/graspPoints_"+"\
# qibullet/grasp_qibullet_1318grasp_points_RGripper.json"
min_quality = 0
min_time = 1

mainGripper(object_file, grasp_points_path_file_simox, min_time, min_quality,
False)

# grasp_points = [[[-0.07302517, -0.04576785, 0.02571655], [0.044, 0.029, 0.242, 0.969]]]
# gripper_name = "rGripper"
# grasping(object_file, gripper_name, grasp_points, min_time, min_quality)

print("Duration:", time.time()-start_all)
