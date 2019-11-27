import math
from lxml import etree
import json
import numpy as np
from numpy.linalg import inv

PATH_DATA = "data/"
PATH_JSON = PATH_DATA + "JSON/"
PATH_XML = PATH_DATA + "XML/"
PATH_OBJECT_DATA = PATH_DATA + "object_data/"
METRIC_TRESHOLD = 0.49


def quaternionFromMatrix(matrix):
    """
    Transform a matrix of transformation or rotation in quaternion

    Parameters:
        matrix - the matrix to transform

    Returns:
        quaternion - quaternion
    """
    round_number = 3
    w = math.sqrt(max(0, 1 + matrix[0][0] + matrix[1][1] + matrix[2][2])) / 2
    x = math.sqrt(max(0, 1 + matrix[0][0] - matrix[1][1] - matrix[2][2])) / 2
    y = math.sqrt(max(0, 1 - matrix[0][0] + matrix[1][1] - matrix[2][2])) / 2
    z = math.sqrt(max(0, 1 - matrix[0][0] - matrix[1][1] + matrix[2][2])) / 2
    if(x * (matrix[2][1] - matrix[1][2]) < 0):
        x = -x

    if(y * (matrix[0][2] - matrix[2][0]) < 0):
        y = -y

    if(z * (matrix[1][0] - matrix[0][1]) < 0):
        z = -z

    quaternion = [round(x, round_number), round(y, round_number),
                  round(z, round_number), round(w, round_number)]

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
    grasp_points["parameters"]["frame"] = "object_centroid"
    grasp_points["parameters"]["gripper"] = {}
    grasp_points['grasps'] = []
    end_effector = ""
    round_number = 3
    # Need milimeters
    unity = 1000.0
    for grasp in tree.xpath("/ManipulationObject/GraspSet"):
        end_effector = str(grasp.get("EndEffector"))
        grasp_points["parameters"]["gripper"] = end_effector
    count = 0
    for grasp in tree.xpath("/ManipulationObject/GraspSet/Grasp"):
        eef_in_center_frame = []
        quality = grasp.get("quality")
        
        if float(quality) > METRIC_TRESHOLD:
            count += 1
            tree_matrix = next(grasp.iter("TransformGlobalPose")).getchildren()[0]\
                .getchildren()
            eef_in_center_frame.append(
                [round(float(tree_matrix[0].get("c1")), round_number),
                round(float(tree_matrix[0].get("c2")), round_number),
                round(float(tree_matrix[0].get("c3")), round_number),
                round(float(tree_matrix[0].get("c4")), round_number)])

            eef_in_center_frame.append(
                [round(float(tree_matrix[1].get("c1")), round_number),
                round(float(tree_matrix[1].get("c2")), round_number),
                round(float(tree_matrix[1].get("c3")), round_number),
                round(float(tree_matrix[1].get("c4")), round_number)])

            eef_in_center_frame.append(
                [round(float(tree_matrix[2].get("c1")), round_number),
                round(float(tree_matrix[2].get("c2")), round_number),
                round(float(tree_matrix[2].get("c3")), round_number),
                round(float(tree_matrix[2].get("c4")), round_number)])
            
            pos = [round(float(eef_in_center_frame[0][3])/unity,
                        round_number),
                round(float(eef_in_center_frame[1][3])/unity,
                        round_number),
                round(float(eef_in_center_frame[2][3])/unity,
                        round_number)]
            grasp_points['grasps'].append(
                [float(quality), pos, quaternionFromMatrix(eef_in_center_frame)
                ])
    grasp_points['grasps'] = sorted(grasp_points["grasps"])
    print("-", count, "grasp points")
    return grasp_points

def main(xml_file_name_grasp_points, file_name_save):
    """
    Create a JSON file containing the name of the end effector, the quality of
    each grasp point, the 6D position in quaternion and milimeters

    Parameters:
        xml_file_name_grasp_points - xml file returned by Simox containing
        grasp_points
        file_name_save - the name of the JSON file created
    """
    grasp_points_rotation_matrix = \
        simoxMatrixTransformationToDict(xml_file_name_grasp_points)

    json_grasp_points = json.dumps(grasp_points_rotation_matrix)

    f = open(PATH_JSON +
             file_name_save + "_" +
             grasp_points_rotation_matrix["parameters"]['gripper']+".json", "w")
    f.write(json_grasp_points)
    f.close()
    name_file_saved = file_name_save + "_" +\
        grasp_points_rotation_matrix["parameters"]['gripper']+".json"
    print("File created:", name_file_saved)

    return name_file_saved
