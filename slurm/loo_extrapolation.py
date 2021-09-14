import os
from utils import start_script, train_command, eval_command, clean_command, create_command


cities = [19, 27, 34, 50, 77, 78, 84, 99]

def make_slurm(station, exp_name, job_name_short, job_name_long, seq_len, d_model, nlayers, alpha_mixup):
    pattern = f"{station}TRAIN"
    val_pattern = f"{station}VAL"
    start = start_script(job_name=job_name_short)
    stations_train = [c for c in cities if c != station]
    cmd_create = create_command(station=stations_train,
                                name=station,
                                station_val=station)
    cmd1 = train_command(name=exp_name,
                         pattern=pattern,
                         val_pattern=val_pattern,
                         seq_len=seq_len,
                         d_model=d_model,
                         layers=nlayers,
                         other_args=[f"--alpha_mixup {alpha_mixup}"])
    cmd2 = eval_command(name=exp_name,
                        train_pattern=pattern,
                        seq_len=seq_len,
                        d_model=d_model,
                        layers=nlayers)
    cmd3 = clean_command(job_name_long=job_name_long,
                         exp_name=exp_name,
                         #other_args=[
                         #    f"rm data/regression/HUR/HUR_{pattern}.csv",
                         #    f"rm data/regression/HUR/HUR_{val_pattern}.csv",
                         #    ]
                         )
    with open(job_name_long, "w") as fh:
        fh.write(start)
        fh.write(cmd_create)
        fh.write(cmd1)
        fh.write(cmd2)
        fh.write(cmd3)


if __name__ == "__main__":
    nlayers = 4
    d_model = 128
    seq_len = 30
    d_ff = 128
    alpha_mixup = 0.2
    for c in cities:
        exp_name = f"{c}_extrapolation_{alpha_mixup}"
        job_name_long = f"slurm/{alpha_mixup}_{c}_140921.slurm"
        make_slurm(station=c,
                   exp_name=exp_name,
                   job_name_short=f"{c}_left_{alpha_mixup}",
                   job_name_long=job_name_long,
                   seq_len=seq_len,
                   d_model=d_model,
                   nlayers=nlayers,
                   alpha_mixup=alpha_mixup
                   )

        os.system(f"sbatch {job_name_long}")
