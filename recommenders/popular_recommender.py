import polars as pl
import pandas as pd 
import numpy as np


def calculate_popularity(df_pd: pd.DataFrame, last_n: int = 100) -> pd.DataFrame:

    df_pl = pl.from_pandas(df_pd.reset_index())
    df_pl = df_pl.sort(['user_id', 'timestamp'])
    df_pl = df_pl.with_column(
        pl.col('item_id')
        .cumcount(reverse=True)
        .over(['user_id'])
        .alias('rank')
        )
    
    df_pl = df_pl.filter(pl.col('rank') < last_n)
    popularity = df_pl.groupby('item_id').agg(
        pl.col('timespent')
        .mean()
        .alias('mean_timespent')
        ).sort(['mean_timespent'], reverse=True)
    
    return popularity.to_pandas()


class PopularRecommender:
    def __init__(self, items_popularity: pd.DataFrame):
        self.items_popularity = items_popularity

    def recommend(
            self, history_df: pd.DataFrame, 
            n_recs: int = 100, 
            filter_items: np.ndarray = None) -> pd.DataFrame:
        
        if filter_items is not None:
            most_popular = self.items_popularity[~self.items_popularity.item_id.isin(filter_items)].iloc[:n_recs]
        else:
            most_popular = self.items_popularity.iloc[:n_recs]

        merge_key_val = 1
        merge_key_col = 'key'
        unique_users = pl.DataFrame({'user_id': history_df.user_id.unique()})
        unique_users = unique_users.with_column(pl.lit(merge_key_val).alias(merge_key_col))

        most_popular = pl.from_pandas(most_popular)
        most_popular = most_popular.with_column(pl.lit(merge_key_val).alias(merge_key_col))

        candidates = unique_users.join(most_popular, on=merge_key_col)
        candidates = candidates.join(
            pl.from_pandas(history_df), on=['user_id', 'item_id'], how='anti').drop(merge_key_col)
        
        return candidates.to_pandas()