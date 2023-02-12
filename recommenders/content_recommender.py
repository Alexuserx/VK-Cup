import os
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from recommenders.utils import (_get_most_relevant_el,
                                _get_most_relevant_el_nz,
                                _get_most_relevant_els,
                                _get_most_relevant_els_nz)


def calculate_similar_items(
        embeddings_matrix: np.ndarray, 
        candidates_df: np.ndarray = None,        
        n_recs: int = 200, 
        save_result: bool = False,
        file_name: str = 'embed', 
        root_dir: str = '../models/content/') -> pd.DataFrame:
    
    if candidates_df is not None:
        start_index = 0
        embeddings_matrix_to_search = embeddings_matrix[candidates_df.item_id.values]
    else:
        start_index = 1
        embeddings_matrix_to_search = embeddings_matrix.copy()

    def calc_similarity(item):
        sim_values = embeddings_matrix[item] @ embeddings_matrix_to_search.T
        best_idxs = (-sim_values).argsort()[start_index:n_recs+1]

        return pd.DataFrame.from_dict({
            'item_id': np.ones_like(best_idxs) * item, 
            'content_sim_rec': best_idxs, 
            'content_sim_score': sim_values[best_idxs], 
            'content_sim_rank': range(len(best_idxs))
            })

    similarities = Parallel(n_jobs=-1)(
        delayed(calc_similarity)(item) for item in tqdm(range(embeddings_matrix.shape[0])))
    
    similarities = pd.concat(similarities).reset_index(drop=True)

    if candidates_df is not None:
        similarities['content_sim_rec'] = similarities['content_sim_rec'].map(
            dict(zip(candidates_df.index.values, candidates_df.item_id.values)))

        similarities = similarities[similarities['item_id'] != similarities['content_sim_rec']]

        new_rank = [range(group) for group in similarities.groupby('item_id')['content_sim_rank'].agg(len)]
        new_rank = [el for items in new_rank for el in  items]
        similarities['content_sim_rank'] = new_rank        

    if save_result:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        similarities_path = os.path.join(root_dir, f'similarities_{file_name}.parquet.gzip')
        
        similarities.to_parquet(similarities_path, compression='gzip')

    return similarities


class ContentRecommender:
    def __init__(self, similarities_df: pd.DataFrame = None, similarities_filepath: str = None):
        if similarities_df is not None:
            self.similarities_df = similarities_df.reset_index(drop=True)
        elif similarities_filepath is not None:
            self.similarities_df = pd.read_parquet(similarities_filepath).reset_index(drop=True)
        else:
            raise ValueError('Neither dataframe nor filepath given')

    def _get_recs_for_one(self, candidates: pd.DataFrame, user_id: int) -> pd.DataFrame:
        return pd.DataFrame.from_dict({
            'user_id': np.ones_like(candidates['content_sim_rec']) * user_id,
            'item_id': candidates['content_sim_rec'],
            'content_sim_score': candidates['content_sim_score'],
            'content_sim_rank': candidates['content_sim_rank']
            })

    def _get_recs_for_one_multiple(self, candidates: pd.DataFrame, user_id: int) -> pd.DataFrame:
        recs_len = len(candidates.iloc[0]['content_sim_rec'])
        recs_num = candidates.shape[0]
        return pd.DataFrame.from_dict({
            'user_id': np.ones(recs_len * recs_num, dtype=np.int32) * user_id,
            'item_id': [el for rec in candidates['content_sim_rec'] for el in rec],
            'content_sim_score': [el for score in candidates['content_sim_score'] for el in score],
            'content_sim_rank': [el for rank in candidates['content_sim_rank'] for el in rank],
            })

    def recommend(self, users_history: pd.DataFrame, n_recs: int = 200, mode = 'MNZ', **kwargs) -> pd.DataFrame:
        max_els = kwargs.get('max_els', 2)
        n_recs = n_recs if 'M' not in mode else n_recs // max_els
        modes_dict = {
            'Z': _get_most_relevant_el,
            'NZ': _get_most_relevant_el_nz,
            'MZ': partial(_get_most_relevant_els, max_els=max_els),
            'MNZ': partial(_get_most_relevant_els_nz, max_els=max_els)
            }
        get_relevant_item = modes_dict.get(mode, None)
        get_recs = self._get_recs_for_one if 'M' not in mode else self._get_recs_for_one_multiple
        print(f'Recommender with args: mode={mode}, n_recs={n_recs}, max_els={max_els}')
        
        users_history = users_history.groupby('user_id', as_index=False)[['item_id', 'timespent']].agg(list)
    
        similarities_cands = self.similarities_df[
            self.similarities_df['content_sim_rank'] < n_recs].groupby('item_id').agg(list)

        recommendations_list = []
        for user_id, user_items, user_timespent in tqdm(
                zip(users_history['user_id'], users_history['item_id'], users_history['timespent']),
                total=len(users_history)):

            most_relevant_item = get_relevant_item(user_items, user_timespent)
            candidates = similarities_cands.iloc[most_relevant_item]

            recs = get_recs(candidates, user_id)
            recommendations_list.append(recs[~recs['item_id'].isin(user_items)])

        return pd.concat(recommendations_list).reset_index(drop=True)


def get_content_similarity_features(
        embeddings_matrix: np.ndarray, 
        candidates_df: pd.DataFrame,
        history_df: pd.DataFrame) -> pd.DataFrame:

    print(f'Users candidates df w shape: {candidates_df.shape[0]:_}')
    print(f'Users history df w shape: {history_df.shape[0]:_}')

    users_candidates = candidates_df.groupby('user_id', as_index=False)['item_id'].agg(list)
    users_history = history_df.groupby('user_id', as_index=False)[['item_id', 'timespent']].agg(list)

    print(f'Users candidates list len: {len(users_candidates):_}')
    print(f'Users history list len: {len(users_history):_}')
    
    features_list = []
    for user_id, user_candidates, user_history, user_timespent in tqdm(
            zip(users_candidates['user_id'], 
                users_candidates['item_id'], 
                users_history['item_id'], 
                users_history['timespent']),
            total=len(users_history)
            ):

        # calculate features using only most relevant items in history
        # user_history = _get_most_relevant_els(user_history, user_timespent, max_els=4)

        similarities = embeddings_matrix[user_candidates] @ embeddings_matrix[user_history].T

        features = pd.DataFrame.from_dict({
            'user_id': np.ones_like(user_candidates, dtype=np.int32) * user_id,
            'item_id': user_candidates,

            'content_sim_mean': similarities.mean(axis=1),
            'content_sim_min': similarities.min(axis=1),
            'content_sim_max': similarities.max(axis=1),
            'content_sim_std': similarities.std(axis=1),
             })

        features_list.append(features)

    return pd.concat(features_list).reset_index(drop=True)