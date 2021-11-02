import os
from extrapolation import make_slurm

cities = [15, 19, 27, 34, 50, 54, 77, 78, 84, 85, 99]

if __name__ == "__main__":
    dic = {15: (2, 1, 2, 0, 64, 30),
 19: (64, 4, 2, 0.1, 64, 60),
 27: (4, 4, 2, 0.1, 32, 60),
 34: (16, 4, 8, 0.1, 64, 60),
 50: (16, 2, 8, 0.1, 64, 60),
 54: (4, 4, 2, 0.1, 32, 60),
 77: (8, 2, 8, 0.1, 32, 60),
 78: (32, 4, 2, 0.1, 64, 60),
 84: (16, 4, 2, 0.1, 64, 60),
 85: (16, 4, 2, 0.1, 64, 60),
 99: (16, 4, 2, 0.1, 64, 60)}
    epochs = 2000
    batch_size = 32
    data_dir = "data"
    ncpus = 2
    res_dir = "res"
    for city in cities:
        (d, n_l, n_h, do, L, ff)  = dic[city]
        exp_name = f"{city}_retrain"
        job_name_long = f"slurm/{city}_retrain.slurm"

        make_slurm(station=city,
                   exp_name=exp_name,
                   job_name_short=f"{city}",
                   job_name_long=job_name_long,
                   seq_len=L,
                   d_model=d,
                   d_ff=ff,
                   nlayers=n_l,
                   heads=n_h,
                   batch_size=batch_size,
                   epochs=epochs,
                   ncpus=ncpus,
                   )
        os.system(f"sbatch {job_name_long}")
