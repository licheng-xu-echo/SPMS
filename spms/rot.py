import numpy as np

def pt2axis(point,axis=[0,0,1]):
    ## Rotates a point to align with the coordinate axis, obtaining the corresponding rotation matrix, as determined by the 'axis' parameter.
    assert axis in [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],'wrong axis parameter'
    point = point / np.linalg.norm(point)
    rot_axis = np.cross(point, np.array(axis))
    rot_axis = rot_axis / np.linalg.norm(rot_axis)  
    axis_max = max(axis)
    if axis_max == 1:
        angle = np.arccos(point[np.argmax(axis)])
    else:
        angle = np.arccos(-point[np.argmin(axis)])
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def pt2plane_around_axis(point,axis='z',plane='xz'):
    # Rotate a point around a given coordinate axis to align with a specified plane.

    assert axis in ['x','y','z'], 'wrong axis parameter'
    assert plane in ['xy','xz','yz'], 'wrong plane parameter'
    assert axis in plane, 'wrong axis parameter'

    if axis == 'x':
        if plane == 'xy' or plane == 'yx':
            # Rotate around x-axis to lay on the xy plane
            angle = -np.arctan2(point[2], point[1])
        else:  # 'xz' plane
            # Rotate around x-axis to lay on the xz plane
            angle = np.arctan2(point[1], point[2])
    elif axis == 'y':
        if plane == 'xy' or plane == 'yx':
            # Rotate around y-axis to lay on the xy plane
            angle = np.arctan2(point[2], point[0])
        else:  # 'yz' plane
            # Rotate around y-axis to lay on the yz plane
            angle = -np.arctan2(point[0], point[2])
    else:  # 'z' axis
        if plane == 'xz' or plane == 'zx':
            # Rotate around z-axis to lay on the xz plane
            angle = -np.arctan2(point[1], point[0])
        else:  # 'yz' plane
            # Rotate around z-axis to lay on the yz plane
            angle = np.arctan2(point[0], point[1])
    # Create the corresponding rotation matrix
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        rotation_matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # 'z' axis
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    return rotation_matrix

def rotate_round_axis(axis='z', angle_degrees=180):
    # Rotate a point around a given coordinate axis by a specified angle.
    angle_radians = np.radians(angle_degrees)
    
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    return rotation_matrix

def tript2fix_orit(positions,originpt,axispt,planept,axis='z',plane='xz'):

    # Fix the orientation of the molecule by anchoring three points.
    axis_map = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]}
    # To origin
    originpt = np.mean([positions[pt] for pt in originpt],axis=0)
    positions = positions - np.array(originpt)

    # Rotate to positive axis
    axispt = np.mean([positions[pt] for pt in axispt],axis=0)
    rot2axis_mat = pt2axis(axispt,axis=axis_map[axis])
    positions = np.array([np.dot(rot2axis_mat, pos) for pos in positions])
    # Rotate to plane
    planept = np.mean([positions[pt] for pt in planept],axis=0)
    rot2plane_mat = pt2plane_around_axis(planept,axis=axis,plane=plane)
    positions = np.array([np.dot(rot2plane_mat, pos) for pos in positions])
    return positions