import os
from utils import start_script, train_command, eval_command, clean_command
import itertools

d_model = [16, 64]

def make_slurm(exp_name, job_name_short, job_name_long, seq_len, d_model, val_contiguous=False, nlayers=2):
    start = start_script(job_name=job_name_short)
    if val_contiguous:
        other_args = ["--val_continuous"]
    else:
        other_args = None
    cmd1 = train_command(name=exp_name, pattern="77TRAIN", seq_len=seq_len, layers=nlayers, other_args=other_args)
    cmd2 = eval_command(name=exp_name, train_pattern="77TRAIN", seq_len=seq_len, d_model=d_model, layers=nlayers)
    cmd3 = clean_command(job_name_long=job_name_long, exp_name=exp_name)
    with open(job_name_long, "w") as fh:
        fh.write(start)
        fh.write(cmd1)
        fh.write(cmd2)
        fh.write(cmd3)


if __name__ == "__main__":
    nlayers = [2, 4]
    d_model = [16, 64, 256]
    seq_len = [10, 30]
    val_contiguous = [True, False]
    for i, (d, s, v, n) in enumerate(itertools.product(d_model, seq_len, val_contiguous, nlayers)):
        exp_name = f"77_{d}_{s}_{n}_val_c" if v else f"77_{d}_{s}_{n}_random_val"
        job_name_long = f"slurm/extrapolation_77_{i}.slurm"
        make_slurm(exp_name=exp_name, job_name_short="extra77", job_name_long=job_name_long, seq_len=s, d_model=d, val_contiguous=v, nlayers=n)
        os.system(f"sbatch {job_name_long}")
