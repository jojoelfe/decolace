from pyquaternion import Quaternion 
from scipy.spatial.transform import Rotation
import numpy as np
import eulerangles


def calculate_quaternion_rotation(angle1, angle2):
    rot0 = Rotation.from_euler('ZYZ', -angle1, degrees=True)
    rot1 = Rotation.from_euler('ZYZ', -angle2, degrees=True)
    # Convert to quaternions
    q0 = Quaternion([rot0.as_quat()[3], rot0.as_quat()[0], rot0.as_quat()[1], rot0.as_quat()[2]])
    q1 = Quaternion([rot1.as_quat()[3], rot1.as_quat()[0], rot1.as_quat()[1], rot1.as_quat()[2]])
    # rotation that converts q1 into q0
    return q1 * q0.inverse

def calculate_angle_between_vectors(angle1, angle2, vector):
    rot1 = eulerangles.invert_rotation_matrices(eulerangles.euler2matrix(angle1, axes='zyz',
                             intrinsic=True,
                             right_handed_rotation=True))
    rot2 = eulerangles.invert_rotation_matrices(eulerangles.euler2matrix(angle2, axes='zyz',
                             intrinsic=True,
                             right_handed_rotation=True))

    rotated_vector1 = np.dot(rot1, vector)
    rotated_vector2 = np.dot(rot2, vector)

    unit_vector1 = rotated_vector1 / np.linalg.norm(rotated_vector1)
    unit_vector2 = rotated_vector2 / np.linalg.norm(rotated_vector2)

    dot_product = np.dot(unit_vector1, unit_vector2)

    angle = np.arccos(dot_product)
    return (np.degrees(angle), unit_vector1, unit_vector2)

