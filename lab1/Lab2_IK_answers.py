import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入:
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    max_iterations = 100
    tolerance = 0.01
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    initial_position = meta_data.joint_initial_position
    path, pathname, path_endTo, path_rootTo = meta_data.get_path_from_root_to_end()
    joint_num = len(path)
    joint_num_1 = len(path_endTo)
    joint_num_2 = len(path_rootTo)
    joint_rotation = np.zeros((len(joint_name), 4))
    flag_done = False
    for iterations in range(max_iterations):
        # 从end开始遍历path1上的点
        for i in range(1, joint_num_1):
            # 计算旋转轴和角度
            vec_to_end = normalize(joint_positions[path[-1]] - joint_positions[path_endTo[i]])
            vec_to_target = normalize(target_pose - joint_positions[path_endTo[i]])
            cos_theta = np.dot(vec_to_target, vec_to_end)
            sin_theta = np.linalg.norm(np.cross(vec_to_end, vec_to_target))
            theta = 0.2 * np.arctan2(sin_theta, cos_theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            rotation_axis = normalize(np.cross(vec_to_end, vec_to_target))
            if theta < 0.0001:
                continue
            for j in range(i, -1, -1):
                # 更新路径上后继关节的位置和旋转
                vec_to_joint = joint_positions[path_endTo[j]] - joint_positions[path_endTo[i]]
                vec_to_joint_update = cos_theta * vec_to_joint + (1 - cos_theta) * np.dot(vec_to_joint, rotation_axis) * rotation_axis + sin_theta * np.cross(rotation_axis, vec_to_joint)
                joint_positions[path_endTo[j]] = joint_positions[path_endTo[i]] + vec_to_joint_update
                rotation_quat = R.from_rotvec(rotation_axis * theta).as_quat()
                quat_current = joint_orientations[path_endTo[j]]
                quat_update = R.from_quat(rotation_quat) * R.from_quat(quat_current)
                joint_orientations[path_endTo[j]] = quat_update.as_quat()
                if joint_rotation[path_endTo[j]].all() == 0:
                    joint_rotation[path_endTo[j]] = R.from_quat(rotation_quat).as_quat()
                else:
                    update_rotation = R.from_quat(rotation_quat) * R.from_quat(joint_rotation[path_endTo[j]])
                    joint_rotation[path_endTo[j]] = update_rotation.as_quat()
            if np.linalg.norm(target_pose - joint_positions[path[-1]]) < tolerance:
                flag_done = True
                break
        if flag_done:
            break
        # 从root开始反向遍历path2上的点
        for i in range(joint_num_2, 0, -1):
            vec_to_end = normalize(joint_positions[path[-1]] - joint_positions[path_rootTo[i - 1]])
            vec_to_target = normalize(target_pose - joint_positions[path_rootTo[i - 1]])
            cos_theta = np.dot(vec_to_target, vec_to_end)
            sin_theta = np.linalg.norm(np.cross(vec_to_end, vec_to_target))
            theta = 0.2 * np.arctan2(sin_theta, cos_theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            rotation_axis = normalize(np.cross(vec_to_end, vec_to_target))
            if theta < 0.0001:
                continue
            for j in range(i, joint_num):
                vec_to_joint = joint_positions[path[j]] - joint_positions[path_rootTo[i - 1]]
                vec_to_joint_update = cos_theta * vec_to_joint + (1 - cos_theta) * np.dot(vec_to_joint, rotation_axis) * rotation_axis + sin_theta * np.cross(rotation_axis, vec_to_joint)
                joint_positions[path[j]] = joint_positions[path_rootTo[i - 1]] + vec_to_joint_update
                rotation_quat = R.from_rotvec(rotation_axis * theta).as_quat()
                quat_current = joint_orientations[path[j]]
                quat_update = R.from_quat(rotation_quat) * R.from_quat(quat_current)
                joint_orientations[path[j]] = quat_update.as_quat()
                if joint_rotation[path[j]].all() == 0:
                    joint_rotation[path[j]] = R.from_quat(rotation_quat).as_quat()
                else:
                    update_rotation = R.from_quat(rotation_quat) * R.from_quat(joint_rotation[path[j]])
                    joint_rotation[path[j]] = update_rotation.as_quat()
            if np.linalg.norm(target_pose - joint_positions[path[-1]]) < tolerance:
                flag_done = True
                break
        if flag_done:
            break
    for i in range(len(joint_name)):
        if i in path:
            continue
        if joint_parent[i] == -1:
            continue
        if not joint_rotation[joint_parent[i]].all() == 0:
            update_orientation = R.from_quat(joint_rotation[joint_parent[i]]) * R.from_quat(joint_orientations[i])
            joint_orientations[i] = update_orientation.as_quat()
            joint_rotation[i] = R.from_quat(joint_rotation[joint_parent[i]]).as_quat()
            offset = initial_position[i] - initial_position[joint_parent[i]]
            parent_orientation = R.from_quat(joint_orientations[joint_parent[i]])
            offset = parent_orientation.apply(offset)
            joint_positions[i] = joint_positions[joint_parent[i]] + offset

    # print(iterations)
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    lshoulder, lelbow, lwrist, lwristend
    9,         10,     11,     12
    """
    root_position = joint_positions[0]
    target_position = root_position.copy()
    target_position[0] += relative_x
    target_position[2] += relative_z
    target_position[1] = target_height
    max_iterations = 100
    tolerance = 0.01
    flag_done = False
    for iteration in range(max_iterations):
        for i in range(11, 8, -1):
            vec_to_end = normalize(joint_positions[12] - joint_positions[i])
            vec_to_target = normalize(target_position - joint_positions[i])
            cos_theta = np.dot(vec_to_target, vec_to_end)
            sin_theta = np.linalg.norm(np.cross(vec_to_end, vec_to_target))
            theta = 0.2 * np.arctan2(sin_theta, cos_theta)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            rotation_axis = normalize(np.cross(vec_to_end, vec_to_target))
            if theta < 0.0001:
                continue
            for j in range(i, 13):
                vec_to_joint = joint_positions[j] - joint_positions[i]
                vec_to_joint_update = cos_theta * vec_to_joint + (1 - cos_theta) * np.dot(vec_to_joint, rotation_axis) * rotation_axis + sin_theta * np.cross(rotation_axis, vec_to_joint)
                joint_positions[j] = joint_positions[i] + vec_to_joint_update
                rotation_quat = R.from_rotvec(rotation_axis * theta).as_quat()
                quat_current = joint_orientations[j]
                quat_update = R.from_quat(rotation_quat) * R.from_quat(quat_current)
                joint_orientations[j] = quat_update.as_quat()
            if np.linalg.norm(target_position - joint_positions[12]) < tolerance:
                flag_done = True
                break
        if flag_done:
            break
    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations
