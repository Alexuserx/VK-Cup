
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse


def convert_to_sparse_ui(data: pd.DataFrame) -> sparse.csr_matrix:
    sparse_matrix = sparse.csr_matrix((
        data['timespent'].astype(float).values,
        (data['user_id'].values, data['item_id'].values)))      
    return sparse_matrix


def recalculate_target_users_hist(
        df_pd: pd.DataFrame, 
        alpha_coeff: float = 40.0) -> pd.DataFrame:
    
    """
    Recalculate timespent based on item's rank in user history
    expression: timespent = 1 + alpha * timespent / (1 + rank), 
    where rank of last viewed item equals 0
    """

    df_pl = pl.from_pandas(df_pd.reset_index())
    df_pl = df_pl.sort(['user_id', 'timestamp'])
    df_pl = df_pl.with_column(
        pl.col('item_id')
        .cumcount(reverse=True)
        .over(['user_id'])
        .alias('rank')
        )
    df_pd = df_pl.to_pandas()
    df_pd['timespent'] = 1 + ((alpha_coeff * df_pd['timespent'].ge(0)) / (1 + df_pd['rank']))
    df_pd.drop(['rank'], axis=1, inplace=True)
    return df_pd.set_index('timestamp')


# Marshalkin Nikita's original idea
def recalculate_target_all_time(
        df_pd: pd.DataFrame, 
        alpha_coeff: float = 100.0, 
        reaction_coeff: float = 6.0, 
        decay_power: float = 1e7) -> pd.DataFrame:
    
    """
    Recalculculate timespent using all history
    """

    df_pd['rank'] = df_pd.index.max() - df_pd.index.to_numpy()
    df_pd['timespent'] = 1 + alpha_coeff\
                             * np.exp(-df_pd['rank'] / decay_power)\
                             * (df_pd['timespent'].ge(0) + reaction_coeff * (df_pd['reaction'] > 0))
    df_pd.drop(['rank'], axis=1, inplace=True)
    return df_pd

def convert_to_sparse_ui(data: pd.DataFrame) -> sparse.csr_matrix:
    sparse_matrix = sparse.csr_matrix((
        data['timespent'].astype(float).values,
        (data['user_id'].values, data['item_id'].values)))      
    return sparse_matrix


def fix_dtypes(df_pl: pl.DataFrame) -> pl.DataFrame:

    for col in ['user_id', 'item_id', 'source_id']:
        if col in df_pl.columns:
            df_pl = df_pl.with_columns(pl.col(col).cast(pl.Int32))

    df_pl = df_pl.with_columns([
        pl.col(col).cast(pl.Float32) for col in df_pl.columns\
        if df_pl[col].dtype == pl.Float64 or
        (df_pl[col].dtype == pl.Int64 and df_pl[col].null_count() > 0)
        ])
    
    if '__index_level_0__' in df_pl.columns:
        df_pl = df_pl.drop(['__index_level_0__'])   

    return df_pl


def _get_most_relevant_el_nz(user_items: List, user_timespent: List) -> int:
    user_items_np = np.array(user_items)
    user_timespent = np.array(user_timespent)
    user_timespent_nz_idxs = np.nonzero(user_timespent)
    user_timespent_nz = user_timespent[user_timespent_nz_idxs]
    user_rels = user_timespent_nz / range(1, 1 + len(user_timespent_nz))[::-1]
    return user_items_np[user_timespent_nz_idxs][user_rels.argmax()]\
            if len(user_timespent_nz_idxs[0]) > 0 else user_items[-1:]

def _get_most_relevant_el(user_items: List, user_timespent: List) -> int:
    user_items = np.array(user_items)
    user_timespent = np.array(user_timespent)
    user_rels = user_timespent / range(1, 1 + len(user_timespent))[::-1]
    return user_items[user_rels.argmax()]

def _get_most_relevant_els_nz(user_items: List, user_timespent: List, max_els: int = 3) -> np.ndarray:
    user_items_np = np.array(user_items)
    user_timespent = np.array(user_timespent)
    user_timespent_nz_idxs = np.nonzero(user_timespent)
    user_timespent_nz = user_timespent[user_timespent_nz_idxs]
    user_rels = user_timespent_nz / range(1, 1 + len(user_timespent_nz))[::-1]
    return user_items_np[user_timespent_nz_idxs][(-user_rels).argsort()[:max_els]]\
            if len(user_timespent_nz_idxs[0]) > 0 else user_items[-min(len(user_items), max_els):]

def _get_most_relevant_els(user_items: List, user_timespent: List, max_els: int = 3) -> int:
    user_items = np.array(user_items)
    user_timespent = np.array(user_timespent)
    user_rels = user_timespent / range(1, 1 + len(user_timespent))[::-1]
    return user_items[(-user_rels).argsort()[:max_els]]