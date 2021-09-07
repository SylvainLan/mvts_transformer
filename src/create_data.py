import pandas as pd
import argparse


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities_test", nargs="+", type=int, required=True)
    parser.add_argument("--cities_val", nargs="+", type=int, required=True)
    parser.add_argument("--name", type=str, required=False, default="")
    parser.add_argument("--cities_train", nargs="+", type=int, required=False, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse()
    cities_train = args.cities_train
    cities_test = args.cities_test
    cities_val = args.cities_val
    name = args.name

    if name == "":
        name = f"{city_val}{city_test}"

    df = pd.read_csv("data/regression/HUR/HUR_FULL.csv")
    cities = df.CITY_NAME.unique()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["CITY_NAME", "date"])
    if cities_train is None:
        cities_train = list(set(cities).difference([*cities_test, *cities_val]))
    df_train = df.loc[cities_train]
    df_val = df.loc[cities_val]

    df_train.to_csv(f"data/regression/HUR/HUR_{name}TRAIN.csv")
    df_val.to_csv(f"data/regression/HUR/HUR_{name}VAL.csv")
