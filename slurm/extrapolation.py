import os
from utils import start_script, train_command, eval_command, clean_command, create_command
import itertools


def make_slurm(station, exp_name, job_name_short, job_name_long, seq_len, d_model, val_contiguous=False, nlayers=2):
    pattern = f"{station}TRAIN"
    start = start_script(job_name=job_name_short)
    if val_contiguous:
        other_args = ["--val_continuous"]
    else:
        other_args = None
    cmd_create = create_command(station=station,
                                name=station,
                                split_dates=True)
    cmd1 = train_command(name=exp_name,
                         pattern=pattern,
                         seq_len=seq_len,
                         d_model=d_model,
                         layers=nlayers,
                         other_args=other_args)
    cmd2 = eval_command(name=exp_name,
                        train_pattern=pattern,
                        seq_len=seq_len,
                        d_model=d_model,
                        layers=nlayers)
    cmd3 = clean_command(job_name_long=job_name_long,
                         exp_name=exp_name,
                         other_args=[f"rm data/regression/HUR/HUR_{pattern}.csv"])
    with open(job_name_long, "w") as fh:
        fh.write(start)
        fh.write(cmd_create)
        fh.write(cmd1)
        fh.write(cmd2)
        fh.write(cmd3)


def make_slurm_split_val(station, exp_name, job_name_short, job_name_long, seq_len, d_model, nlayers=2, n_splits=4):
    pattern = f"{station}TRAIN"
    val_pattern = f"{station}VAL"
    start = start_script(job_name=job_name_short)
    cmd_create = create_command(station=station,
                                name=station,
                                n_splits=n_splits)
    cmds_train = []
    cmds_eval = []
    cmds_clean = []

    job_name_long = job_name_long.split(".slurm")[0]
    jobs_name_long = [f"{job_name_long}_{i}.slurm" for i in range(n_splits)]
    for i in range(n_splits):
        cmd1 = train_command(name=f"{exp_name}{i}",
                             pattern=f"{pattern}{i}",
                             val_pattern=f"{val_pattern}{i}",
                             seq_len=seq_len,
                             d_model=d_model,
                             layers=nlayers)
        cmd2 = eval_command(name=f"{exp_name}{i}",
                            train_pattern=f"{pattern}{i}",
                            seq_len=seq_len,
                            d_model=d_model,
                            layers=nlayers)
        cmd3 = clean_command(job_name_long=jobs_name_long[i],
                             exp_name=f"{exp_name}_{i}",
                             other_args=[f"rm data/regression/HUR/HUR_{pattern}{i}.csv",
                                         f"rm data/regression/HUR/HUR_{val_pattern}{i}.csv"])
        cmds_train.append(cmd1)
        cmds_eval.append(cmd2)
        cmds_clean.append(cmd3)
    for i in range(n_splits):
        with open(jobs_name_long[i], "w") as fh:
            fh.write(start)
            fh.write(cmd_create)
            fh.write(cmds_train[i])
            fh.write(cmds_eval[i])
            fh.write(cmds_clean[i])


if __name__ == "__main__":
    nlayers = 4
    d_model = 64
    seq_len = 30
    d_ff = 64
    n_splits = 4
    cities = [19, 27, 34, 50, 77, 78, 84, 99]
    for c in cities:
        exp_name = f"{c}_extrapolation"
        job_name_long = f"slurm/{c}_130921.slurm"
        make_slurm_split_val(station=c,
                             exp_name=exp_name,
                             job_name_short=f"{c}_extra",
                             job_name_long=job_name_long,
                             seq_len=seq_len,
                             d_model=d_model,
                             nlayers=nlayers,
                             n_splits=n_splits
                             )

        job_name_long = job_name_long.split(".slurm")[0]
        for i in range(n_splits):
            os.system(f"sbatch {job_name_long}_{i}.slurm")
