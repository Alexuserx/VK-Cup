import os
import pickle

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from recommenders.utils import _get_most_relevant_els, convert_to_sparse_ui


def train_als(
        data: pd.DataFrame, 
        factors: int = 512, 
        iterations: int = 100, 
        regularization: float = 0.5,  
        save_result: bool = True,        
        model_name: str = 'als', 
        root_dir: str = '../models/als/'
        ) -> tuple[AlternatingLeastSquares, sparse.csr_matrix]:
    
    recommender = AlternatingLeastSquares(
        factors = factors, 
        iterations = iterations, 
        regularization = regularization
        )

    sparse_user_item = convert_to_sparse_ui(data)
    print(f'Converted data to sparse: {sparse_user_item.shape}')

    recommender.fit(sparse_user_item)

    if save_result:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        model_path = os.path.join(root_dir, f'recommender_{model_name}.pkl')
        matrix_path = os.path.join(root_dir, f'matrix_{model_name}.npz')
        
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        print(f'Saved model to: {model_path}')

        sparse.save_npz(matrix_path, sparse_user_item)
        print(f'Saved sparse matrix to: {matrix_path}')

    return recommender, sparse_user_item


class ALSRecommender:
    def __init__(
            self, als_model: AlternatingLeastSquares,
            sparse_user_item: sparse.csr_matrix = None, 
            train_df: pd.DataFrame = None) -> None:
        self.recommender = als_model
        if sparse_user_item is not None:
            self.sparse_user_item = sparse_user_item
        elif train_df is not None:
            self.sparse_user_item = convert_to_sparse_ui(train_df)
        else:
            raise ValueError('Neither dataframe nor sparse matrix given')
        
    def recommend(
            self, user_ids: np.ndarray, 
            n_recs: int = 100, 
            filter_items: np.ndarray = None) -> pd.DataFrame:
        
        recs, scores = self.recommender.recommend(
            user_ids, 
            self.sparse_user_item[user_ids],
            N=n_recs, 
            filter_already_liked_items=True, 
            filter_items=filter_items
            )    
        
        candidates = pd.DataFrame.from_dict({
            'user_id': np.repeat(user_ids, [len(rec) for rec in recs]),
            'item_id': recs.ravel(),
            'als_sim_score': scores.ravel(),
            'als_sim_rank': [rank for rec in recs for rank, _ in enumerate(rec)],
            }) 
        
        return candidates
    

def get_als_similarity_features(
        als_model: AlternatingLeastSquares, 
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

        similarities = cosine_similarity(als_model.item_factors[user_candidates], 
                                         als_model.item_factors[user_history])

        # recalculate als similarity score and rank for all candidates - 
        # initially score was calculated only for als candidates
        als_sim_score = (als_model.user_factors[[user_id]]\
                         @ als_model.item_factors[[user_candidates]][0].T)[0]

        features = pd.DataFrame.from_dict({
            'user_id': np.ones_like(user_candidates, dtype=np.int32) * user_id,
            'item_id': user_candidates,
            'als_sim_mean': similarities.mean(axis=1),
            'als_sim_min': similarities.min(axis=1),
            'als_sim_max': similarities.max(axis=1),
            'als_sim_std': similarities.std(axis=1),
            'als_sim_score': als_sim_score,
            'als_sim_rank': (-als_sim_score).argsort()
             })

        features_list.append(features)

    return pd.concat(features_list).reset_index(drop=True)
    