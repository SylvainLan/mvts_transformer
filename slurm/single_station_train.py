import os
import subprocess
import argparse

from utils import start_script, train_command, eval_command, clean_command, create_command


def train(station,
          exp_name,
          job_name_long,
          seq_len,
          d_model,
          d_ff,
          nlayers,
          heads,
          batch_size,
          n_splits,
          epochs,
          ncpus):
    pattern = f"{station}TRAIN"
    val_pattern = f"{station}VAL"

    cmds_train = []
    cmds_eval = []
    cmds_clean = []
    for i in range(n_splits):
        cmd1 = train_command(name=f"{exp_name}_{i}",
                             pattern=f"{pattern}{i}",
                             val_pattern=f"{val_pattern}{i}",
                             seq_len=seq_len,
                             d_model=d_model,
                             d_ff=d_ff,
                             layers=nlayers,
                             heads=heads,
                             batch_size=batch_size,
                             epochs=epochs,
                             stdout=False)
        cmd2 = eval_command(name=f"{exp_name}_{i}",
                            train_pattern=f"{pattern}{i}",
                            seq_len=seq_len,
                            batch_size=batch_size,
                            d_model=d_model,
                            d_ff=d_ff,
                            heads=heads,
                            layers=nlayers,
                            eval_pattern="TRAIN",
                            stdout=False)
        #cmd3 = clean_command(job_name_long=job_name_long,
        #                     exp_name=f"{exp_name}_{i}",
        #                     )
        cmds_train.append(cmd1)
        cmds_eval.append(cmd2)
        #cmds_clean.append(cmd3)

    for i in range(n_splits):
        subprocess.run(cmds_train[i])
        subprocess.run(cmds_eval[i])
        #subprocess.run(cmds_clean[i])


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_station", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse()
    index_station = args.index_station
    seq_len = args.seq_len
    d_model = args.d_model
    d_ff = args.d_ff
    heads = args.heads
    layers = args.layers
    dropout = args.dropout


    batch_size = 32
    ncpus = 2
    n_splits = 3
    epochs = 10
    cities = [15, 19, 27, 34, 50, 54, 77, 78, 84, 85, 99]

    station = cities[index_station]
    exp_name = f"{d_model}_{layers}_{heads}_{dropout}_{layers}_{d_ff}"
    job_name_long = f"slurm/{station}_{exp_name}.slurm"
    train(station=station,
          exp_name=exp_name,
          job_name_long=job_name_long,
          seq_len=seq_len,
          d_model=d_model,
          d_ff=d_ff,
          nlayers=layers,
          heads=heads,
          batch_size=batch_size,
          n_splits=3,
          epochs=epochs,
          ncpus=ncpus)
