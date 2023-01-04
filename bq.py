import plotly.graph_objs as go
import cv2
import numpy as np
from libsvm import svmutil  # used in brisque
from brisque import BRISQUE as bq
from math import isnan
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# TODO: This implementation of the brisque metric is very slow.
#       Should update following the opencv implementation.
#       https://learnopencv.com/image-quality-assessment-brisque/
#       see also:
#       https://github.com/krshrimali/No-Reference-Image-Quality-Assessment-using-BRISQUE-Model


def scoreCube(cube: np.ndarray) -> np.ndarray:
    """Calculate the brisque metric.

    Args:
        cube (np.ndarray): The cube to be evaluated.

    Returns:
        np.ndarray: The brisque metric for each wavelength.
    """
    brisque_metric = np.zeros(cube.shape[2])
    for i in range(cube.shape[2]):
        brisque_metric[i] = scoreImage(cube[:, :, i])

    return brisque_metric


def scoreImage(image: np.ndarray) -> float:
    """Calculate the brisque metric.

    Args:
        image (np.ndarray): The image to be evaluated.

    Returns:
        float: The value in terms of the brisque metric.
    """

    s = bq()

    ret = s.get_score(image)

    if isnan(ret):
        ret = 100

    return ret


def plotAllScores(wavelengths: np.ndarray, ScoreMatrix: np.ndarray, titles: list) -> go.Figure:
    """Plot all scores.

    Args:
        wavelengths (np.ndarray)[1xM]: The Wavelengths.
        ScoreMatrix (np.ndarray)[NxM]: The score matrix. with N scenes and M wavelengths.
        titles (list(str)): The titles for the legend of the line plots.
    Returns:
        None: Will plot the BRISQUE scores.
    """
    fig = go.Figure()

    for i in range(ScoreMatrix.shape[0]):
        fig.add_trace(go.Scatter(
            name=titles[i],
            x=wavelengths,
            y=ScoreMatrix[i, :],
            mode='lines',
        ))

    fig.update_layout(
        title="All BRISQUE scores",
        yaxis_title='BRISQUE Value',
        hovermode="x"
    )

    # fig.show()
    return fig


def plotScoreStatistics(wavelengths: np.ndarray,
                        scoreMatrix: np.ndarray,
                        VarianceScale: float = 1.0,
                        title: str = "") -> go.Figure:
    """Plot statistics from scoring scores.
    Args:
        wavelengths (np.ndarray)[1xM]: The Wavelengths.
        scoreMatrix (np.ndarray)[NxM]: The score matrix. with N scenes and M wavelengths.
        VarianceScale (float): Scale the variance.
        title (str, optional): The title of the plot.

    Returns:
        None: Will plot the BRISQUE scores.
    """

    mean = np.mean(np.asarray(scoreMatrix), axis=0)
    median = np.median(np.asarray(scoreMatrix), axis=0)
    var = np.var(np.asarray(scoreMatrix), axis=0)
    ind_max = np.unravel_index(np.argmax(mean, axis=None), mean.shape)
    ind_min = np.unravel_index(np.argmin(mean, axis=None), mean.shape)

    fig = go.Figure([
        go.Scatter(
            name='Mean',
            x=wavelengths,
            y=mean,
            mode='lines',
        ),
        go.Scatter(
            name='Median',
            x=wavelengths,
            y=median,
            mode='lines',
        ),
        go.Scatter(
            name='Max',
            x=[wavelengths[ind_max]],
            y=[mean[ind_max]],
            mode='markers',
        ),
        go.Scatter(
            name='Min',
            x=[wavelengths[ind_min]],
            y=[mean[ind_min]],
            mode='markers',
        ),
        go.Scatter(
            name='Upper Bound',
            x=wavelengths,
            y=mean+var*VarianceScale,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=wavelengths,
            y=mean-var*VarianceScale,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig.update_layout(
        title=title,
        yaxis_title='BRISQUE score',
        hovermode="x"
    )

    # fig.show()
    return fig


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("image.jpg")

    # Compute the BRISQUE score
    # score = compute_brisque_score(image)
    # # Print the score
    # print("BRISQUE score:", score)

    score = scoreImage(image)
    print("BRISQUE score:", score)
