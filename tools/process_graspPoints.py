import math
from scipy.spatial.transform import Rotation as R
from lxml import etree
import json
import numpy as np
from numpy.linalg import inv

### Matrix of transformation ###
marker_in_center_frame =  np.array([
                            [1.0, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, 0, 1.0, 30],
                            [0, 0, 0, 1.0]])
#Right hand
gripper_in_wrist_frame = np.array([
                        [1, 0, 0, 25],
                        [0, 0.8186398, -0.5743073, 0],
                        [0, 0.5743073,  0.8186398, 0],
                        [0, 0, 0, 1]])
                        
eef_in_gripper_frame = np.array([
                    [0.7071068,  0, 0.7071068, 25],
                    [0.50, -0.7071068, -0.5, -15],
                    [0.50,  0.7071068, -0.5, -25],
                    [0, 0, 0, 1]])

inv_wrist_in_eef_frame = inv(np.matmul(gripper_in_wrist_frame, \
eef_in_gripper_frame))

def quaternionFromMatrix(matrix):
    """
    Transform a matrix of transformation or rotation in quaternion

    Parameters:
        matrix - the matrix to transform

    Returns:
        quaternion - quaternion
    """
    round_number = 3
    w = math.sqrt( max( 0, 1 + matrix[0][0] + matrix[1][1] + matrix[2][2])) / 2
    x = math.sqrt( max( 0, 1 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2
    y = math.sqrt( max( 0, 1 - matrix[0][0] + matrix[1][1] - matrix[2][2])) / 2
    z = math.sqrt( max( 0, 1 - matrix[0][0] - matrix[1][1] + matrix[2][2])) / 2
    if(x * ( matrix[2][1] - matrix[1][2] ) < 0 ):
        x = -x

    if(y * ( matrix[0][2] - matrix[2][0] ) < 0 ):
        y = -y

    if(z * ( matrix[1][0] - matrix[0][1] ) < 0 ):
        z = -z

    quaternion = [round(x,round_number), round(y,round_number), \
    round(z,round_number), round(w,round_number)]

    return quaternion

def simoxMatrixTransformationToDict(file_path):
    """
    Parse a XML file returned by Simox and create a dict
    Convert the coordinates of grasp points in wrist referencial

    Parameters:
        file_path - The path of Simox's xml file

    Returns:
        grasp_points - Dict with the quality and 6D grasp point,
        position and rotation matrix
    """

    tree = etree.parse(file_path)
    grasp_points = {}
    grasp_points["parameters"] = {}
    grasp_points["parameters"]["gripper"] = {}
    grasp_points['grasps'] = {}
    end_effector = ""
    round_number = 3
    #Need milimeters
    unity = 1000.0
    for grasp in tree.xpath("/ManipulationObject/GraspSet"):
        end_effector = str(grasp.get("EndEffector"))
        grasp_points["parameters"]["gripper"] = end_effector

    for grasp in tree.xpath("/ManipulationObject/GraspSet/Grasp"):
        eef_in_center_frame = []
        quality = grasp.get("quality")
        if float(quality) not in grasp_points['grasps'].keys():
            grasp_points['grasps'][float(quality)] = []

        tree_matrix = next(grasp.iter("TransformGlobalPose")).getchildren()[0]\
            .getchildren()
        eef_in_center_frame.append([round(float(tree_matrix[0].get("c1")),
        round_number), round(float(tree_matrix[0].get("c2")),round_number),
        round(float(tree_matrix[0].get("c3")),round_number),
        round(float(tree_matrix[0].get("c4")),round_number)])

        eef_in_center_frame.append([round(float(tree_matrix[1].get("c1")),
        round_number), round(float(tree_matrix[1].get("c2")),round_number),
        round(float(tree_matrix[1].get("c3")),round_number),
        round(float(tree_matrix[1].get("c4")),round_number)])

        eef_in_center_frame.append([round(float(tree_matrix[2].get("c1")),
        round_number), round(float(tree_matrix[2].get("c2")),round_number),
        round(float(tree_matrix[2].get("c3")),round_number),
        round(float(tree_matrix[2].get("c4")),round_number)])

        matrix_wrist_in_center_frame = \
        np.matmul(eef_in_center_frame, inv_wrist_in_eef_frame)

        pos=[round(float(matrix_wrist_in_center_frame[0][3])/unity,round_number)
        , round(float(matrix_wrist_in_center_frame[1][3])/unity,round_number),
        round(float(matrix_wrist_in_center_frame[2][3])/unity,round_number)]

        grasp_points['grasps'][float(quality)].append(
        [pos,matrix_wrist_in_center_frame])

    return grasp_points

def graspPointsToQuaternion(grasp_points_rotation_matrix):
    """
    For all grasp point, change the rotate matrix to quaternion.
    Stock the new dict in a JSON file.

    Parameters:
        grasp_points_rotation_matrix - Dict which contains the quality, the pos
        and the transform matrix of each grasp points

    Returns:
        grasp_points_quaternion - Dict  which contains the quality, the pos and
        the quaternion of each grasp points
    """
    grasp_points_quaternion = {}
    grasp_points_quaternion["parameters"] = {}
    grasp_points_quaternion["parameters"]["gripper"] = \
    grasp_points_rotation_matrix["parameters"]['gripper']
    grasp_points_quaternion["grasps"] = {}
    qualities = grasp_points_rotation_matrix["grasps"].keys()
    count = 0
    for quality in qualities:
        grasp_points = grasp_points_rotation_matrix["grasps"][quality]

        if quality not in grasp_points_quaternion["grasps"].keys():
            grasp_points_quaternion["grasps"][quality] = []

        for grasp in grasp_points:
            count += 1
            grasp_points_quaternion["grasps"][quality].append([
            grasp[0],quaternionFromMatrix(grasp[1])])
    print("-",count,"grasp points")
    return grasp_points_quaternion

def tabascoToQibullet(tabasco_grasp_point):
    """
    Transform a grasp point in tabasco referential in qibullet referential

    Parameters:
        tabasco_grasp_point - the 6D grasp point in tabasco referential

    Returns:
        qibullet_grasp_point - the 6D grasp point in qibullet referential
    """
    gripper_in_wrist_frame[:3,-1] *= 0.001

    marker_in_center_frame[:3,-1] *= 0.001

    gripper_in_marker_frame = tabasco_grasp_point
    rWrist_in_gripper_frame = inv(gripper_in_wrist_frame)


    transform_matrix = (marker_in_center_frame.dot(
    gripper_in_marker_frame)).dot(rWrist_in_gripper_frame)
    qibullet_grasp_point = quaternionFromMatrix(transform_matrix)

    return [transform_matrix[:2,-1], qibullet_grasp_point]


def main(xml_file_name_grasp_points, file_name_save):
    """
    Create a JSON file containing the name of the end effector, the quality of
    each grasp point, the 6D position in quaternion and milimeters

    Parameters:
        xml_file_name_grasp_points - xml file returned by Simox containing grasp
        grasp_points
        file_name_save - the name of the JSON file created
    """
    grasp_points_rotation_matrix = \
    simoxMatrixTransformationToDict(xml_file_name_grasp_points)
    grasp_points_quaternion = \
    graspPointsToQuaternion(grasp_points_rotation_matrix)

    json_grasp_points = json.dumps(grasp_points_quaternion)

    f = open("../qibullet/graspPoints_data/JSON/graspPoints_simoxCgal/"
    +file_name_save+"_"+
    grasp_points_quaternion["parameters"]['gripper']+".json","w")
    f.write(json_grasp_points)
    f.close()
    print("File created:", file_name_save+"_"+
    grasp_points_quaternion["parameters"]['gripper']+".json")

    return


xml_file_name_grasp_points="../qibullet/graspPoints_data/XML/1grasp_points.xml"
file_name_save="1grasp_points"

# main(xml_file_name_grasp_points, file_name_save)

# old grasp point
grasp_tabasco = np.array([
                        [0.881007, -0.341898,  0.324806,  -0.051],
                        [0.470714,  0.679377,  -0.561642,  -0.034],
                        [-0.028662,  0.648166,  0.760017,  -0.005],
                        [0, 0, 0, 1],])

# grasp_tabasco = np.array([
                            # [0.7823135169,  -0.484133327,  0.395716857,  -0.033456175],
                            # [0.4102259212,  -0.08094618268,  -0.9100295363,  -0.0837517],
                            # [0.4719021114,  0.8729570351,  0.1350768671,  -0.0082109],
                            # [0,  0,  0,  1],])



print("New grasp:", tabascoToQibullet(grasp_tabasco))
