import numpy as np

from constant.face_analysis import DistanceMetric


def compute_cosine_distance(
    source_representation: np.ndarray | list,
    test_representation: np.ndarray | list,
) -> np.float64:
    """
    Compute cosine distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def compute_euclidean_distance(
    source_representation: np.ndarray | list,
    test_representation: np.ndarray | list,
) -> np.float64:
    """
    Compute euclidean distance between two given vectors
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated euclidean distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(
        np.multiply(euclidean_distance, euclidean_distance),
    )
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x: np.ndarray | list) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
    Returns:
        y (np.ndarray): l2 normalized vector
    """
    if isinstance(x, list):
        x = np.array(x)
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def compute_distance(
    alpha_embedding: np.ndarray | list,
    beta_embedding: np.ndarray | list,
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
) -> np.float64:
    """
    Wrapper to compute distance between vectors according to the given distance metric
    Args:
        source_representation (np.ndarray or list): 1st vector
        test_representation (np.ndarray or list): 2nd vector
    Returns
        distance (np.float64): calculated cosine distance
    """
    if distance_metric == DistanceMetric.COSINE:
        distance = compute_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == DistanceMetric.EUCLIDEAN:
        distance = compute_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == DistanceMetric.EUCLIDEAN_L2:
        distance = compute_euclidean_distance(
            l2_normalize(alpha_embedding),
            l2_normalize(beta_embedding),
        )
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return distance
