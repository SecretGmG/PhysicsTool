import numpy as np
import matplotlib.pyplot as plt


def matrix_quiver(
    x: np.ndarray,
    y: np.ndarray,
    matrices: np.ndarray,
    shade_determinant=False,
    label=None,
    det_label="determinant",
):
    """
    Visualizes eigenvectors of matrices using quiver plots.

    Args:
        x (np.ndarray): X-coordinates for the quiver plot.
        y (np.ndarray): Y-coordinates for the quiver plot.
        matrices (np.ndarray): Matrices for which eigenvectors are computed.
        shade_determinant (bool): If True, shades the plot based on the determinant of the matrices.
        label (str): Label for the eigenvectors. optional defaults to None
        det_label (str): Label for the determinant. optional defaults to 'determinant'
    Returns:
        None
    """
    # remove the arrowheads and set the pivot to mid to vizualize eigenvectors
    EIGEN_VEC_QUIVER_KWARGS = {
        "headwidth": 0,
        "headlength": 0,
        "headaxislength": 0,
        "pivot": "mid",
    }

    eigenvalues, eigenvectors = np.linalg.eig(matrices)
    # from the docs of np.linalg.eig : eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i] !!!

    # scale by the eigenvalues
    scaled_eigenvectors = eigenvectors * eigenvalues[..., None, :]
    determinants = eigenvalues[..., 0] * eigenvalues[..., 1]

    if shade_determinant:
        mesh = plt.pcolormesh(x, y, determinants)
        plt.colorbar(mesh, label=det_label)

    q1 = plt.quiver(
        x,
        y,
        scaled_eigenvectors[..., 0, 0],
        scaled_eigenvectors[..., 1, 0],
        **EIGEN_VEC_QUIVER_KWARGS,
    )
    q2 = plt.quiver(
        x,
        y,
        scaled_eigenvectors[..., 0, 1],
        scaled_eigenvectors[..., 1, 1],
        **EIGEN_VEC_QUIVER_KWARGS,
    )

    # ensure correct relative scaling of the eigenvector arrows
    scale = np.max(q1.scale, q2.scale)
    q1.scale = scale
    q2.scale = scale

    if label is not None:
        # add empty scatter plot to add label, is a little hacky but works
        plt.scatter(None, None, marker="+", label=label, color="black")
