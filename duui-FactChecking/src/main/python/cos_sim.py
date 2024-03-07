from tqdm import tqdm
from scipy.spatial import distance
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity


def list_cos_sim(list_vec1: List[List[Union[float, int]]], list_vec2: List[List[Union[float, int]]]) -> List[float]:
    cos_sim_list = []
    for c, list_i in tqdm(enumerate(list_vec1), desc=f"Compute all cos sim"):
        cos_sim_list.append(compute_cos_sim(list_i, list_vec2[c]))
    return cos_sim_list


def list_cos_sim_sklearn(list_vec1: List[List[Union[float, int]]], list_vec2: List[List[Union[float, int]]]) -> List[float]:
    return cosine_similarity(list_vec1, list_vec2).tolist()


def compute_cos_sim(vec1: List[Union[float, int]], vec2: List[Union[float, int]]) -> float:
    cos_sim = 1 - distance.cosine(vec1, vec2)
    return cos_sim
