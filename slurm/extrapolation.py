import os
import argparse
from utils import start_script, train_command, eval_command, clean_command, create_command
import itertools
import glob


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
               ncpus,
               n_splits=1):
    pattern = f"{station}TRAIN"
    start = start_script(job_name=job_name_short, mode="cpu", ncpus=ncpus)
    cmd_create = create_command(station=station,
                                name=station)
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
                         other_args=[f"rm data/regression/HUR/HUR_{pattern}.csv"])
    with open(job_name_long, "w") as fh:
        fh.write(start)
        fh.write(cmd_create)
        fh.write(cmd1)
        fh.write(cmd2)
        fh.write(cmd3)


def make_slurm_split_val(station,
                         exp_name,
                         job_name_short,
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
    start = start_script(job_name=job_name_short, mode="cpu", ncpus=ncpus)
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
                             d_ff=d_ff,
                             layers=nlayers,
                             heads=heads,
                             batch_size=batch_size,
                             epochs=epochs)
        cmd2 = eval_command(name=f"{exp_name}{i}",
                            train_pattern=f"{pattern}{i}",
                            seq_len=seq_len,
                            batch_size=batch_size,
                            d_model=d_model,
                            d_ff=d_ff,
                            heads=heads,
                            layers=nlayers)
        cmd3 = clean_command(job_name_long=jobs_name_long[i],
                             exp_name=f"{exp_name}{i}",
                             )
        cmds_train.append(cmd1)
        cmds_eval.append(cmd2)
        cmds_clean.append(cmd3)
    for i in range(n_splits):
        with open(jobs_name_long[i], "w") as fh:
            fh.write(start)
            #fh.write(cmd_create)
            fh.write(cmds_train[i])
            fh.write(cmds_eval[i])
            fh.write(cmds_clean[i])



#def _parse():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--ncpus", type=int, default=1)
#    parser.add_argument("--nlayers", type=int, default=2)
#    parser.add_argument("--heads", type=int, default=4)
#    parser.add_argument("--d_model", type=int, default=32)
#    parser.add_argument("--seq_len", type=int, default=30)
#    parser.add_argument("--d_ff", type=int, default=64)
#    parser.add_argument("--batch_size", type=int, default=32)
#    parser.add_argument("--exp_prefix", type=str, default="")
#    parser.add_argument("--n_splits", type=int, default=1)
#    args = parser.parse_args()
#    return args


if __name__ == "__main__":

    # args = _parse()
    # ncpus = args.ncpus
    # nlayers = args.nlayers
    # heads = args.heads
    # d_model = args.d_model
    # seq_len = args.seq_len
    # d_ff = args.d_ff
    # batch_size = args.batch_size
    # n_splits = args.n_splits
    #exp_prefix = args.exp_prefix
    #if exp_prefix != "":
    #    exp_prefix = f"{exp_prefix}_"

    hidden_dim = [2]
    n_layers = [1]
    heads = [2]
    dropout = [.1]
    seq_len = [60]
    d_ff = [2]
    epochs = 2000
    batch_size = 32
    n_splits = 3
    ncpus = 2
    exp_prefix = "cv"


    cities = [19, 27, 34, 50, 54, 77, 78, 84, 85, 99]

    if n_splits > 1:
        slurmer = make_slurm_split_val
    else:
        slurmer = make_slurm


    for i, (d, n_l, n_h, do, L, f) in enumerate(itertools.product(hidden_dim, n_layers, heads, dropout, seq_len, d_ff)):
        for c in cities:
            exp_name = f"{exp_prefix}{c}_extrapolation"
            job_name_long = f"slurm/{exp_prefix}{c}_261021.slurm"
            slurmer(station=c,
                    exp_name=exp_name,
                    job_name_short=f"{c}_extra",
                    job_name_long=job_name_long,
                    seq_len=L,
                    d_model=d,
                    d_ff=f,
                    nlayers=n_l,
                    heads=n_h,
                    batch_size=batch_size,
                    n_splits=n_splits,
                    epochs=epochs,
                    ncpus=ncpus,
                    )

            job_name_long = job_name_long.split(".slurm")[0]
            for f in glob.glob(f"{job_name_long}*.slurm"):
                os.system(f"sbatch {f}")
