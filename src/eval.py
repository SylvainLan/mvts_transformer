"""
Written by George Zerveas

If you use any part of the code in this repository, please consider citing the following paper:
George Zerveas et al. A Transformer-based Framework for Multivariate Time Series Representation Learning, in
Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14--18, 2021
"""

import os
import sys
import time
import pickle
import json
import pandas as pd

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Project modules
from options import Options
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets.data import data_factory, Normalizer
from datasets.datasplit import split_dataset
from models.ts_transformer import model_factory
from models.loss import get_loss_module
from optimizers import get_optimizer


def main(config):
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    data_class = data_factory[config['data_class']]
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'], limit_size=config['limit_size'], config=config)
    feat_dim = my_data.feature_df.shape[1]  # dimensionality of data features

    # Split dataset
    test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
    train_indices = my_data.all_IDs
    test_indices = test_data.all_IDs

    # Pre-process features
    normalizer = None
    normalizer = Normalizer(config['normalization'])
    my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
    test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    model = model_factory(config, my_data)
    # Load model and optimizer state
    if args.load_model:
        model = utils.load_model(model, config['load_model'])

    model.to(device)

    dataset_class, collate_fn, _ = pipeline_factory(config)
    test_dataset = dataset_class(test_data, test_indices)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             pin_memory=True,
                             collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
    dic_idx = test_data.dic_idx
    df = {"CITY_NAME": [],
          "date": [],
          "HUR": [],
          "inferred": [],
          }
    model = model.eval()
    bs = config["batch_size"]
    for i, data in enumerate(test_loader):
        X, targets, padding_masks, IDs = data
        out = model(X.to(device), padding_masks.to(device))
        for j in range(out.size()[0]):
            idx = bs*i + j
            df["CITY_NAME"].append(dic_idx[idx][0])
            df["date"].append(dic_idx[idx][1])
            df["HUR"].append(targets.detach().cpu().numpy()[j][0])
            df["inferred"].append(out.detach().cpu().numpy()[j][0])
    df = pd.DataFrame(df)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["CITY_NAME", "date"])
    df.to_csv(f"{config['output_dir']}/inference.csv")


if __name__ == '__main__':
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)
