import os
from utils import start_script, train_command, eval_command, clean_command, create_command


cities = [19, 27, 34, 50, 77, 78, 84, 99]

def make_slurm(station,
               exp_name,
               job_name_short,
               job_name_long,
               seq_len,
               d_model,
               d_ff,
               nlayers,
               heads,
               batch_size,
               epochs,
               alpha_mixup):
    pattern = f"{station}TRAIN"
    val_pattern = f"{station}VAL"
    start = start_script(job_name=job_name_short, mode="gpu", ncpus=2)
    stations_train = [c for c in cities if c != station]
    cmd_create = create_command(station=stations_train,
                                name=station,
                                station_val=station)
    cmd1 = train_command(name=exp_name,
                         pattern=pattern,
                         val_pattern=val_pattern,
                         seq_len=seq_len,
                         d_model=d_model,
                         d_ff=d_ff,
                         layers=nlayers,
                         heads=heads,
                         batch_size=batch_size,
                         epochs=epochs
                         #other_args=[f"--alpha_mixup {alpha_mixup}"]
                         )
    cmd2 = eval_command(name=exp_name,
                        train_pattern=pattern,
                        seq_len=seq_len,
                        d_model=d_model,
                        d_ff=d_ff,
                        heads=heads,
                        layers=nlayers,
                        batch_size=batch_size,
                        )
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
    nlayers = 2
    d_model = 32
    seq_len = 30
    d_ff = 64
    heads = 4
    alpha_mixup = 0
    batch_size = 32
    epochs = 1000
    for c in cities:
        exp_name = f"{c}_extrapolation"
        job_name_long = f"slurm/{c}_240921.slurm"
        make_slurm(station=c,
                   exp_name=exp_name,
                   job_name_short=f"{c}_extra",
                   job_name_long=job_name_long,
                   seq_len=seq_len,
                   d_model=d_model,
                   d_ff=d_ff,
                   nlayers=nlayers,
                   heads=heads,
                   batch_size=batch_size,
                   epochs=epochs,
                   alpha_mixup=alpha_mixup
                   )

        os.system(f"sbatch {job_name_long}")
