# J'ai pas besoin de test pcq je dois le faire à la toute fin

import pandas as pd
import argparse
import numpy as np


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--cities_train", nargs="+", type=int, required=True)
    parser.add_argument("--cities_val", nargs="+", type=int, required=False, default=None)
    parser.add_argument("--split_dates", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse()
    cities_train = args.cities_train
    cities_val = args.cities_val
    name = args.name
    split_dates = args.split_dates

    if name == "":
        name = f"{city_val}{city_test}"

    df = pd.read_csv("data/regression/HUR/HUR_FULL.csv")
    cities = df.CITY_NAME.unique()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["CITY_NAME", "date"])

    df_train = df.loc[cities_train]
    if cities_val is not None:
        df_val = df.loc[cities_val]
        # TODO
    if split_dates:
        if len(cities_train) > 1:
            raise NotImplementedError("Pas encore implémenté")
        c = cities_train[0]
        dates = df_train.loc[c].index
        for i, d in enumerate(np.array_split(dates, 2)):
            df_train.swaplevel().drop(d).to_csv(f"data/regression/HUR/HUR_{name}TRAIN{i}.csv")
            df_train.swaplevel().loc[d].to_csv(f"data/regression/HUR/HUR_{name}VAL{i}.csv")
    else:
        df_train.to_csv(f"data/regression/HUR/HUR_{name}TRAIN.csv")
