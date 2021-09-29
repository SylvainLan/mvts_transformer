import os
import argparse
from utils import start_script, train_command, eval_command, clean_command, create_command
import itertools
import glob

def make_slurm_ms(exp_name, job_name_long, job_name_short, seq_len, d_model, d_ff, nlayers, heads, batch_size, epochs, ncpus, n_splits=1):
    pattern = "BISTRAIN"
    start = start_script(job_name=job_name_short, mode="gpu", ncpus=ncpus)
    cmd1 = train_command(name=exp_name,
                         pattern=pattern,
                         seq_len=seq_len,
                         d_model=d_model,
                         d_ff=d_ff,
                         layers=nlayers,
                         heads=heads,
                         batch_size=batch_size,
                         epochs=epochs)
    cmd2 = eval_command(name=exp_name,
                        train_pattern=pattern,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        d_model=d_model,
                        d_ff=d_ff,
                        heads=heads,
                        layers=nlayers)
    cmd3 = clean_command(job_name_long=job_name_long,
                         exp_name=exp_name,
                         )
    with open(job_name_long, "w") as fh:
        fh.write(start)
        fh.write(cmd1)
        fh.write(cmd2)
        fh.write(cmd3)

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncpus", type=int, default=1)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--exp_prefix", type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = _parse()

    ncpus = args.ncpus
    nlayers = args.nlayers
    heads = args.heads
    d_model = args.d_model
    seq_len = args.seq_len
    d_ff = args.d_ff
    batch_size = args.batch_size

    exp_prefix = args.exp_prefix
    if exp_prefix != "":
        exp_prefix = f"{exp_prefix}_"

    epochs = 1000
    n_splits = 1

    exp_name = f"{exp_prefix}_multistat"
    job_name_long = f"slurm/multistat{exp_prefix}_290921.slurm"
    make_slurm_ms(
                  exp_name=exp_name,
                  job_name_short=f"multistat",
                  job_name_long=job_name_long,
                  seq_len=seq_len,
                  d_model=d_model,
                  d_ff=d_ff,
                  nlayers=nlayers,
                  heads=heads,
                  batch_size=batch_size,
                  n_splits=n_splits,
                  epochs=epochs,
                  ncpus=ncpus,
                  )

    job_name_long = job_name_long.split(".slurm")[0]
    for f in glob.glob(f"{job_name_long}*.slurm"):
        os.system(f"sbatch {f}")
