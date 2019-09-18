import pybullet as p
import numpy as np
import json
import time
from qibullet import SimulationManager
from pepperGripper_virtual import PepperGripperVirtual
import argparse
from os import path
import tools.process_graspPoints as process_gp
import qibullet.tools as qibullet_tools


class PointGraspSimulationManager(SimulationManager):
    def __init__(self):
        pass

    def spawnGripperPepper(
            self,
            gripper,
            physics_client,
            translation=[0, 0, 0.5],
            quaternion=[0, 0, 0, 1],
            spawn_ground_plane=False,
            useFixedBase=False):
        """
        Loads a Pepper gripper's model in the simulation

        Parameters:
            physics_client - The id of the simulated instance in which the
            robot is supposed to be spawned
            translation - List containing 3 elements, the spawning translation
            [x, y, z] in the WORLD frame
            quaternions - List containing 4 elements, the spawning rotation as
            a quaternion [x, y, z, w] in the WORLD frame
            spawn_ground_plane - If True, the pybullet_data ground plane will
            be spawned

        Returns:
            pepper_virtual - A PepperVirtual object, the Pepper simulated
            instance
        """
        pepper_gripper_virtual = PepperGripperVirtual(gripper)

        if spawn_ground_plane:
            self._spawnGroundPlane(physics_client)

        pepper_gripper_virtual.loadRobot(
            translation,
            quaternion,
            physicsClientId=physics_client,
            useFixedBase=useFixedBase)

        return pepper_gripper_virtual


def mainGripper(object_file, grasp_points_path_file, min_time, min_quality,
                saving=True):
    """
    The end-effector tries to grasp the fixed object using physic properties,
    each grasp point is evaluate and all great ones are stocked in a file
    with its quality.

    Parameters:
        object_file - the object's urdf file in the folder object_data
        grasp_points_path_file - The path to the JSON file containing the valid
        grasp points
        min_time - the minimum time the object has
        not to fall to concider the grasp great
        min_quality - the minimum quality required to consider
        a grasp point great
    """
    f = open(grasp_points_path_file, "r")
    json_data = json.loads(f.read())
    f.close()
    gripper_name = json_data["parameters"]["gripper"]
    if not (gripper_name == "rGripper" or gripper_name == "lGripper"):
        print("Error: No such gripper")
        return

    grasp_points = []
    all_grasp_points = pickGraspPoints(json_data, min_quality)
    for grasp in all_grasp_points:
        pos = grasp[0]
        quaternion = grasp[1]
        grasp_points.append([pos, quaternion])
    if len(all_grasp_points) is 0:
        raise NameError('no grasp points to process')
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

    pos = grasp_point[0]
    quaternion = grasp_point[1]

    cid = p.createConstraint(
            pepper_gripper.robot_model, -1, -1, -1,
            p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0.5])

    # reset position to avoid collision problems
    p.resetBasePositionAndOrientation(
        pepper_gripper.robot_model,
        [0, 0, 0.5], [0, 0, 0, 1])
    time.sleep(.01)

    # Open the hand
    pepper_gripper.setAngles(hand, 1, 0.8)
    time.sleep(.03)

    # Approache of the end-effector towards the object
    # print('Grasp tested:', grasp_point)
    interval = 0.1  # start position
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
            new_pos = moveOnOneAxe(
                0, cid, current_x_pos,
                [pos[0], current_y_pos, current_z_pos+1], quaternion)
            current_x_pos = new_pos[0]
        else:
            x_done = True

        if not (abs(current_y_pos - pos[1]) < precision):
            # print("Move on y")
            new_pos = moveOnOneAxe(1, cid, current_y_pos,
                                   [current_x_pos, pos[1], current_z_pos+1],
                                   quaternion)
            current_y_pos = new_pos[1]
        else:
            y_done = True

        if not (abs(current_z_pos - pos[2]) < precision):
            # print("Move on z")
            new_pos = moveOnOneAxe(2, cid, current_z_pos + 1,
                                   [current_x_pos, current_y_pos, pos[2]],
                                   quaternion)
            current_z_pos = new_pos[2] - 1
        else:
            z_done = True

        if (x_done and y_done and z_done):
            # Close enough
            break

    time.sleep(.1)

    # Close the hand
    pepper_gripper.setAngles(hand, 0, 0.8)
    time.sleep(.3)

    # Evaluation of the grasp
    initial_distance_object_endeffector = qibullet_tools.getDistance(
        p.getBasePositionAndOrientation(object)[0],
        p.getBasePositionAndOrientation(pepper_gripper.robot_model)[0])
    p.removeConstraint(object_constraint)
    time.sleep(.3)

    # Raise the object
    # print("Try to raise the object ")
    current_z_pos = new_pos[2]
    start = time.time()
    while time.time() - start < min_time:
        time.sleep(.01)
        p.setGravity(0, 0, -10)
        z_pivot = [new_pos[0], new_pos[1], current_z_pos]
        p.changeConstraint(cid, z_pivot,
                           jointChildFrameOrientation=quaternion, maxForce=10)
        current_z_pos = current_z_pos + 0.002
    time.sleep(.01)
    final_distance_object_endeffector = qibullet_tools.getDistance(
        p.getBasePositionAndOrientation(object)[0],
        p.getBasePositionAndOrientation(pepper_gripper.robot_model)[0])

    # Evaluation of the grasp point
    quality = evaluteGrasp(initial_distance_object_endeffector,
                           final_distance_object_endeffector)

    p.removeConstraint(cid)
    new_pos[2] = new_pos[2] - 1

    return [quality, [new_pos, quaternion]]


def grasping(object_file, gripper_name, grasp_points, min_time, min_quality=0,
             grasp_points_file=None, saving=False):
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

    simulation_manager = PointGraspSimulationManager()
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
        pepper_gripper = simulation_manager.spawnGripperPepper(
                            gripper_name,
                            client, spawn_ground_plane=True)

        print("-- New grasp:", grasp_nb, '--')
        # print(grasp)
        quality = 0
        pivot = np.array([np.array([0, 0, 0]), np.array([0, 0, 0, 0])])

        # Need to repete the same grasp and average the quality
        # because the physic is not perfect, not always the same
        repetition = 3
        for _ in range(repetition):
            object_constraint = p.createConstraint(
                object, -1, -1, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
            p.resetBasePositionAndOrientation(
                object, [0, 0, 1], [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(
                pepper_gripper.robot_model,
                [0, 0, 0.5], [0, 0, 0, 1])
            grasp_done = oneGrasp(object, object_constraint, pepper_gripper,
                                  gripper_name, grasp, min_time)
            quality += grasp_done[0]
            pivot += grasp_done[1]
        quality /= repetition
        pivot = np.array(pivot) / repetition
        pivot = [list(pivot[0]), list(pivot[1])]

        # Need to remove and create again the model because of physic problems
        p.removeBody(pepper_gripper.robot_model)

        print("The grasp's quality is", float(quality))
        # print("Real grasp point done :", pivot)
        grasp_nb += 1

        if saving:
            # Save great grasp in a file
            quality = float(quality)

            if quality >= float(min_quality):
                if quality not in actual_qualities:
                    actual_qualities.append(quality)
                    great_grasps["grasps"][quality] = []
                great_grasps["grasps"][quality].append(list(pivot))

    if saving:
        f = open(process_gp.PATH_JSON +
                 "grasp_qibullet_" + grasp_points_file + ".json", "w")
        f.write(json.dumps(great_grasps))
        f.close()
        print("File saved:", process_gp.PATH_JSON +
              "grasp_qibullet_" + grasp_points_file + ".json")
    print("All grasp done, saved done")
    time.sleep(3)
    return


def evaluteGrasp(initial_distance_object_endeffector,
                 final_distance_object_endeffector):
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
    if final_distance_object_endeffector - precision >\
            initial_distance_object_endeffector:
        # print("Grasp failed")
        quality = 0
    else:
        # print("Grasp succed")
        quality = 1
    return quality


def pickGraspPoints(json_data, min_quality):
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

    all_grasp_points = json_data['grasps']

    for quality in all_grasp_points:
        if float(quality) >= min_quality:
            for grasp in all_grasp_points[quality]:
                grasp_points.append(grasp)
    print("-------", len(grasp_points), "grasp points !")
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
    step = 0.005  # step to go farword the object
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

    p.changeConstraint(cid, pivot,
                       jointChildFrameOrientation=quaternion, maxForce=30)
    time.sleep(.01)
    return pivot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get all grasp points of a ' +
                                     'xml file and save it on a json file')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='name of the grasp point dataset, a xml file')
    parser.add_argument('--object',
                        type=str,
                        default=None,
                        help='name of the object model file, an urdf file')

    parser.add_argument('--output_file_name',
                        type=str,
                        default="dataset",
                        help='name of the output file, a json file')

    args = parser.parse_args()
    if args.dataset is None or not path.exists(
            process_gp.PATH_XML + args.dataset):
        print("Error: you need to give an available xml file with " +
              "the variable --dataset")
        raise NameError('xml file unvalid')
    if args.object is None or not path.exists(
            process_gp.PATH_OBJECT_DATA + args.object + ".urdf"):
        print("Error: you need to give an available urdf file with " +
              "the variable --object")
        raise NameError('urdf file unvalid')
    xml_file_name_grasp_points =\
        process_gp.PATH_XML + args.dataset

    json_dataset = process_gp.main(xml_file_name_grasp_points,
                                   args.output_file_name)

    start_all = time.time()

    object_file = args.object + ".urdf"
    grasp_points_path_file_qibullet = process_gp.PATH_JSON + json_dataset
    min_quality = 0.5
    min_time = 1

    mainGripper(object_file,
                grasp_points_path_file_qibullet, min_time, min_quality, False)

    print("Duration:", time.time()-start_all)
