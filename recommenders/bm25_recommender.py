import os
import pickle
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import BM25Recommender, ItemItemRecommender
from joblib import Parallel, delayed
from tqdm import tqdm

from recommenders.utils import (_get_most_relevant_el,
                                _get_most_relevant_el_nz,
                                _get_most_relevant_els,
                                _get_most_relevant_els_nz,
                                convert_to_sparse_ui)


def train_bm25(
        data: pd.DataFrame, 
        K: int = 512, 
        K1: float = 1.5, 
        B: float = 0.75,  
        save_result: bool = True,        
        model_name: str = 'bm25', 
        root_dir: str = '../models/bm25/') -> ItemItemRecommender:
    
    recommender = BM25Recommender(K=K, K1=K1, B=B)

    sparse_user_item = convert_to_sparse_ui(data)
    print(f'Converted data to sparse: {sparse_user_item.shape}')

    recommender.fit(sparse_user_item)

    if save_result:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        model_path = os.path.join(root_dir, f'recommender_{model_name}.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        print(f'Saved model to: {model_path}')

    return recommender


class Item2ItemRecommender:
    def __init__(self, similarity_model: ItemItemRecommender):
        self.similarity_model = similarity_model

    def _get_recs_for_one(self, recs, scores, user_id):
        return pd.DataFrame.from_dict({
            'user_id': np.ones_like(recs, dtype=np.int32) * user_id,
            'item_id': recs,
            'bm25_sim_score': scores,
            'bm25_sim_rank': range(recs.shape[0]),
            })

    def recommend(self, users_history: pd.DataFrame, n_recs: int = 200, 
                  filter_items: pd.DataFrame  = None, mode = 'MNZ', **kwargs) -> pd.DataFrame:
        max_els = kwargs.get('max_els', 2)
        n_recs = n_recs if 'M' not in mode else n_recs // max_els
        modes_dict = {
            'Z': _get_most_relevant_el,
            'NZ': _get_most_relevant_el_nz,
            'MZ': partial(_get_most_relevant_els, max_els=max_els),
            'MNZ': partial(_get_most_relevant_els_nz, max_els=max_els)
            }
        get_relevant_item = modes_dict.get(mode, None)
        print(f'Recommender with args: mode={mode}, n_recs={n_recs}, max_els={max_els}')
        
        users_history = users_history.groupby('user_id', as_index=False)[['item_id', 'timespent']].agg(list)

        recommendations_list = []
        for user_id, user_items, user_timespent in tqdm(
                zip(users_history['user_id'], users_history['item_id'], users_history['timespent']),
                total=len(users_history)):

            most_relevant_item = get_relevant_item(user_items, user_timespent)

            if 'M' not in mode:
                scores, recs = self.similarity_model.similar_items(
                    most_relevant_item, N=n_recs, filter_items=filter_items)
                candidates = self._get_recs_for_one(scores, recs, user_id)
            else:
                candidates_list = []
                for _, item_id in enumerate(most_relevant_item):
                    recs, scores = self.similarity_model.similar_items(
                        item_id, N=n_recs, filter_items=filter_items)
                    candidates = self._get_recs_for_one(recs, scores, user_id) 
                    candidates_list.append(candidates)
                candidates = pd.concat(candidates_list)          

            recommendations_list.append(candidates[~candidates['item_id'].isin(user_items)])

        return pd.concat(recommendations_list).drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
    

def get_bm25_similarity_features(
        i2i_model: ItemItemRecommender, 
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
        user_history = _get_most_relevant_els(user_history, user_timespent, max_els=4)

        similarities = (i2i_model.similarity[user_candidates]\
                        @ i2i_model.similarity[user_history].T).toarray()

        features = pd.DataFrame.from_dict({
            'user_id': np.ones_like(user_candidates, dtype=np.int32) * user_id,
            'item_id': user_candidates,

            'bm25_sim_mean': similarities.mean(axis=1),
            'bm25_sim_min': similarities.min(axis=1),
            'bm25_sim_max': similarities.max(axis=1),
            'bm25_sim_std': similarities.std(axis=1),
             })

        features_list.append(features)

    return pd.concat(features_list).reset_index(drop=True)