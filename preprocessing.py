## python preprocessing.py --raw-csv "../data/eComm-behavior/eCommerce-behavior-data-2019-Oct.csv" --data-dir "dataset" 

import os
import argparse
import pickle
import pandas as pd

def sample_raw_data(args):
    raw_df = pd.read_csv(args.raw_csv)
    raw_df = raw_df.dropna()

    temp = raw_df.groupby(
        by=['user_id'], as_index=False).agg({'product_id': 'count'}).rename(
            columns={'product_id': 'total_interactions'})

    raw_df = pd.merge(raw_df, temp, on="user_id", how="left")

    del temp

    sample_df  = raw_df[raw_df.total_interactions.between(200, 600)]

    df1 = sample_df[sample_df.event_type.isin(['cart', 'purchase'])]

    df2 = sample_df[sample_df.event_type.isin(['view'])].sample(df1.shape[0], random_state=1)

    sample_df = pd.concat([df1, df2])
    sample_df.drop_duplicates(subset=['event_time', 'user_id', 'product_id'], inplace=True)
    sample_df['event_time'] = sample_df['event_time'].apply(lambda x : pd.to_datetime(x).value)
    sample_df.reset_index(drop=True, inplace=True)


    del df1, df2

    sample_df[['cat_1', 'cat_2']] = sample_df['category_code'].str.split('.', 1, expand=True)
    sample_df['cat_2'] = sample_df['cat_2'].str.split('.').str[0]

    sample_df['user_id'] = sample_df['user_id'].astype('category')
    sample_df['user_id_num'] = sample_df['user_id'].cat.codes
    user_num_to_id = dict(zip(sample_df['user_id_num'], sample_df['user_id']))

    sample_df['product_id'] = sample_df['product_id'].astype('category')
    sample_df['product_id_num'] = sample_df['product_id'].cat.codes
    item_num_to_id = dict(zip(sample_df['product_id_num'], sample_df['product_id']))

    print(f"Sample data shape {len(sample_df)}")

    print(f"Unique users : {len(user_num_to_id)}, Unique items : {len(item_num_to_id)}")


    with open(os.path.join(args.data_dir, 'user_num_to_id.pkl'), 'wb') as handle:
        pickle.dump(user_num_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.data_dir, 'item_num_to_id.pkl'), 'wb') as handle:
        pickle.dump(item_num_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sample_df.to_csv(os.path.join(args.data_dir, 'sample_data.csv'), index=False)

    print(f"data files save at {args.data_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-csv', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    sample_raw_data(args)
