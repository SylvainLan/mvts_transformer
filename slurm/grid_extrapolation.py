import os
import argparse
from utils import start_script
import itertools
import glob


cities = [15, 19, 27, 34, 50, 54, 77, 78, 84, 85, 99]

def make_slurm(exp_name,
               job_name,
               seq_len,
               d_model,
               d_ff,
               nlayers,
               heads,
               dropout,
               ):
    job_name_short = "grid_search"
    start = start_script(job_name=job_name_short, mode="cpu_long", ncpus=2, ngrid=len(cities))
    command = f"python slurm/single_station_train.py --index_station $SLURM_ARRAY_TASK_ID --seq_len {seq_len} --d_model {d_model} --d_ff {d_ff} --heads {heads} --layers {nlayers} --dropout {dropout}\n"

    with open(job_name, "w") as fh:
        fh.write(start)
        fh.write(command)


if __name__ == "__main__":
    hidden_dim = [2, 8, 32, 128]
    n_layers = [2, 4]
    heads = [1, 4]
    dropout = [.1]
    seq_len = [10, 30, 60]
    d_ff = [32, 64]

    hidden_dim = [4, 16, 64]
    n_layers = [2, 4]
    heads = [1, 4]
    dropout = [.1]
    seq_len = [10, 30, 60]
    d_ff = [32, 64]



    parameters = itertools.product(hidden_dim, n_layers, heads, dropout, seq_len, d_ff)


    for i, (d, n_l, n_h, do, L, ff) in enumerate(parameters):
        if n_h > d:
            print("passing, wrong parameters")
        else:
            exp_name = f"{d}_{n_l}_{n_h}_{do}_{L}_{ff}"
            job_name = f"slurm/{exp_name}.slurm"
            make_slurm(
                       exp_name=exp_name,
                       job_name=job_name,
                       seq_len=L,
                       d_model=d,
                       d_ff=ff,
                       nlayers=n_l,
                       heads=n_h,
                       dropout=do,
                       )

            os.system(f"sbatch {job_name}")
