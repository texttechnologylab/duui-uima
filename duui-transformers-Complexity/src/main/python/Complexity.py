from scipy.spatial import distance
from scipy import linalg
from typing import List, Union
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

import numpy as np

import dcor


def compute_distance_correlation(u: Union[np.ndarray, List[Union[int, float]]],
                                 v: Union[np.ndarray, List[Union[int, float]]]) -> float:
    dis_cor = dcor.distance_correlation(np.array(u), np.array(v))
    return dis_cor


def compute_wasserstein_distance(u: Union[np.ndarray, List[Union[int, float]]],
                                 v: Union[np.ndarray, List[Union[int, float]]]) -> float:
    wasserstein = wasserstein_distance(np.array(u), np.array(v))
    return wasserstein


def compute_jensenshannon_distance(u: Union[np.ndarray, List[Union[int, float]]],
                                   v: Union[np.ndarray, List[Union[int, float]]]) -> float:
    jensen = jensenshannon(np.array(u), np.array(v))
    if np.inf == jensen:
        jensen = 0.0
    return jensen


def compute_bhattacharyya_distance(distribution1: Union[np.ndarray, List[Union[int, float]]], distribution2: Union[np.ndarray, List[Union[int, float]]], ) -> float:
    """ Estimate Bhattacharyya Distance (between General Distributions)

    Args:
        distribution1: a sample distribution 1
        distribution2: a sample distribution 2

    Returns:
        Bhattacharyya distance
    """
    distance_bhattacharyya = 0
    try:
        sq = 0
        for i in range(len(distribution1)):
            sq += np.sqrt(distribution1[i] * distribution2[i])
        if sq == 0:
            distance_bhattacharyya = 0.0
        else:
            distance_bhattacharyya = -np.log(sq)
    except Exception as ex:
        pass
    if np.isnan(distance_bhattacharyya):
        distance_bhattacharyya = 0.0
    return distance_bhattacharyya


def compute_mahalanobis_distance(u: Union[np.ndarray, List[Union[int, float]]],
                                 v: Union[np.ndarray, List[Union[int, float]]]) -> Union[float, np.dtype]:
    try:
        # array_np = np.array([u, v])
        # array_u, array_v = np.meshgrid(u, v)
        # zz = np.c_[array_u.ravel(), array_v.ravel()]
        # mah2d = EmpiricalCovariance().fit(array_np)
        # mah2d_distance = mah2d.mahalanobis(zz)
        # mah2d_2 = Mahalanobis(np.stack((u, v)), 4)
        cov_mat = np.stack((u, v), axis=1)
        cov = np.cov(cov_mat)
        inv_cov = linalg.inv(cov)
        mahalanobis = distance.mahalanobis(u, v, inv_cov)
    except Exception as ex:
        mahalanobis = np.nan
    return mahalanobis


if __name__ == '__main__':
    # array_u = [1, 0, 0]
    # array_v = [0, 1, 0]
    # x = [1.23, 2.12, 3.34, 4.5]
    # y = [2.56, 2.89, 3.76, 3.95]
    # iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    # # compute_distance_correlation(x, y)
    # # compute_mahalanobis_distance(x, y)
    # print(compute_bhattacharyya_distance(array_u, array_v))
    # print(compute_wasserstein_distance(array_u, array_v))
    # print(compute_jensenshannon_distance(array_u, array_v))
    # exit()
    models = {
        # "intfloat/multilingual-e5-base": "intfloat",
        # "setu4993/LEALLA-small": "LEALLA",
        # "facebook/xlm-v-base": "XLMv",
        # "bert-base-multilingual-cased": "BBMC",
        # "sentence-transformers/LaBSE": "LaBSE",
        # "xlm-roberta-large": "XLMRL",
        "Twitter/twhin-bert-large": "TwHIN",
        "cardiffnlp/twitter-xlm-roberta-base": "cardiffnlp",
        # "paraphrase-multilingual-MiniLM-L12-v2": "PMLM12v2",
        # "distiluse-base-multilingual-cased-v2": "DBMCv2"

    }