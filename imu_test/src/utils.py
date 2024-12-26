#!/usr/bin/env python3

import numpy as np

def get_rotation_matrix_x(theta: float) -> np.ndarray:
    """
    X축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix_y(theta: float) -> np.ndarray:
    """
    Y축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix_z(theta: float) -> np.ndarray:
    """
    Z축을 기준으로 회전 행렬을 생성합니다.
    Args:
        theta (float): 회전 각도 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    m_rotation = np.array([
        [cos_theta, -sin_theta, 0, 0],
        [sin_theta, cos_theta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return m_rotation

def get_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    3차원 회전 벡터를 이용하여 3차원 회전 행렬을 생성합니다.
    Args:
        rotation (np.ndarray): [roll, pitch, yaw] 회전 벡터 (라디안)
    Returns:
        np.ndarray: 4x4 회전 행렬
    """
    roll, pitch, yaw = rotation
    m_rotation = get_rotation_matrix_z(yaw) @ get_rotation_matrix_y(pitch) @ get_rotation_matrix_x(roll)
    return m_rotation

def get_translation_matrix(translation: np.ndarray) -> np.ndarray:
    """
    이동 행렬을 생성합니다.
    Args:
        translation (np.ndarray): [x, y, z] 이동 벡터
    Returns:
        np.ndarray: 4x4 이동 행렬
    """
    x, y, z = translation
    m_translation = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    return m_translation