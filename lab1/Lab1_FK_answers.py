import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    """
    joint_name = None
    joint_parent = None
    joint_offset = None
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    stack = []
    current_parent = -1  # 根节点的父关节索引为 -1

    with open(bvh_file_path, 'r') as file:
        lines = file.readlines()

    reading_hierarchy = False
    for line in lines:
        if "HIERARCHY" in line:
            reading_hierarchy = True
        elif "MOTION" in line:
            break  # 结束读取层次结构
        elif reading_hierarchy:
            if "ROOT" in line:
                joint_name.append("RootJoint")
                joint_parent.append(current_parent)
                stack.append(len(joint_name) - 1)
            elif "JOINT" in line:
                joint_name.append(line.split()[-1])
                current_parent = stack[-1]  # 更新当前父关节
                joint_parent.append(current_parent)
                stack.append(len(joint_name) - 1)  # 将当前关节的索引入栈
            elif "End Site" in line:
                current_parent = stack[-1]
                joint_name.append(joint_name[current_parent] + "_end")
                joint_parent.append(current_parent)
                stack.append(len(joint_name) - 1)
                # 对于末端节点，不更新 current_parent
            elif "}" in line:
                stack.pop()
            elif "OFFSET" in line:
                offset_values = line.split()[1:]
                offset_values = [float(value) for value in offset_values]
                joint_offset.append(offset_values)

    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))  # 四元数(x, y, z, w)

    # 获取特定帧的数据
    frame_data = motion_data[frame_id]
    channel_index = 0

    for i in range(len(joint_name)):
        # 根关节的处理
        if joint_parent[i] == -1:
            # 根节点平移 + 旋转
            translation = frame_data[:3]
            rotation = frame_data[3:6]
            channel_index += 6
        else:
            # 其他关节的处理
            translation = joint_offset[i]  # 使用偏移作为局部平移
            if i+1 < len(joint_name) and i == joint_parent[i+1]:  # 非末端关节
                rotation = frame_data[channel_index:channel_index + 3]
                channel_index += 3
            else:  # 末端关节没有旋转
                rotation = [0, 0, 0]

        # 计算旋转四元数
        rot_quat = R.from_euler('XYZ', rotation, degrees=True).as_quat()  # 转换为四元数

        # 应用旋转到偏移上并更新全局位置和旋转
        if joint_parent[i] == -1:
            joint_positions[i] = translation
        else:
            parent_pos = joint_positions[joint_parent[i]]
            parent_orientation = R.from_quat(joint_orientations[joint_parent[i]])
            joint_positions[i] = parent_pos + parent_orientation.apply(translation)

        joint_orientations[i] = rot_quat

        # 对于非根节点，需要将旋转累积到父节点上
        if joint_parent[i] != -1:
            parent_quat = joint_orientations[joint_parent[i]]
            rotation_quat = R.from_quat(parent_quat) * R.from_quat(rot_quat)
            joint_orientations[i] = rotation_quat.as_quat()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    jointA_name, jointA_parent, jointA_offset = part1_calculate_T_pose(A_pose_bvh_path)
    jointT_name, jointT_parent, jointT_offset = part1_calculate_T_pose(T_pose_bvh_path)
    motionA_data = load_motion_data(A_pose_bvh_path)
    frame_num = motionA_data.shape[0]
    jointT_name = [item for item in jointT_name if not item.endswith("_end")]
    jointA_name = [item for item in jointA_name if not item.endswith("_end")]
    motion_data = motionA_data.copy()
    for i in range(len(jointT_name)):
        for j in range(len(jointA_name)):
            if jointA_name[j] == jointT_name[i]:
                for k in range(frame_num):
                    motion_data[k][3 + 3 * i:6 + 3 * i] = motionA_data[k][3 + 3 * j:6 + 3 * j]
                break

    for i in range(len(jointT_name)):
        if jointT_name[i] == "lShoulder":
            for j in range(frame_num):
                motion_data[j][5 + 3 * i] -= 45
        if jointT_name[i] == "rShoulder":
            for j in range(frame_num):
                motion_data[j][5 + 3 * i] += 45
    return motion_data
