# J'ai pas besoin de test pcq je dois le faire à la toute fin

import pandas as pd
import argparse
import numpy as np


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--cities_train", nargs="+", type=int, required=True)
    parser.add_argument("--cities_val", nargs="+", type=int, required=False, default=None)
    parser.add_argument("--n_splits", required=False, type=int, default=1)
    parser.add_argument("--length", required=False, type=int, default=None)
    parser.add_argument("--cities_crop", required=False, nargs="+", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse()
    cities_train = args.cities_train
    cities_val = args.cities_val
    name = args.name
    n_splits = args.n_splits
    length = args.length
    cities_crop = args.cities_crop

    if name == "":
        name = f"{city_val}{city_test}"

    #df = pd.read_csv("data/regression/HUR/HUR_BISTRAIN.csv")
    df = pd.read_csv("data/regression/HUR/HUR_TRAIN.csv")
    cities = df.CITY_NAME.unique()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["CITY_NAME", "date"])

    df_train = df.loc[cities_train]
    if cities_val is not None:
        df_val = df.loc[cities_val]
        df_val.to_csv(f"data/regression/HUR/HUR_{name}VAL.csv")
    if n_splits > 1:
        if len(cities_train) > 1:
            raise NotImplementedError("Pas encore implémenté")
        c = cities_train[0]
        dates = df_train.loc[c].index
        for i, d in enumerate(np.array_split(dates, n_splits)):
            df_train.swaplevel().drop(d).to_csv(f"data/regression/HUR/HUR_{name}TRAIN{i}.csv")
            df_train.swaplevel().loc[d].to_csv(f"data/regression/HUR/HUR_{name}VAL{i}.csv")
    else:
        if length is not None:
            dfs = []
            cities_crop = cities_train if cities_crop is None else cities_crop
            for c in cities_train:
                df_crop = df_train.loc[c]
                if c in cities_crop:
                    index = df_crop.index
                    df_crop = df_crop.loc[index[:length]]
                df_crop["CITY_NAME"] = c
                df_crop = df_crop.reset_index().set_index(["CITY_NAME", "date"]).sort_index()
                dfs.append(df_crop)
            df_train = pd.concat(dfs).sort_index()
        df_train.to_csv(f"data/regression/HUR/HUR_{name}TRAIN.csv")
