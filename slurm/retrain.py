import os
import json
from utils import start_script, train_command, eval_command, clean_command, create_command

cities = [19, 27, 34, 50, 77, 78, 84, 99]

def make_retrain(exp_name, path, station, length, job_name):

    job_name_short = job_name.split("/")[-1].split(".")[0]
    start = start_script(job_name=job_name_short, mode="gpu", ncpus=2)
    dic_config = json.load(open(f"{path}/configuration.json", "r"))
    seq_len = dic_config["max_seq_len"]
    batch_size = dic_config["batch_size"]
    d_model = dic_config["d_model"]
    d_ff = dic_config["dim_feedforward"]
    heads = dic_config["num_heads"]
    nlayers = dic_config["num_layers"]

    pattern = f"{station}RETRAINTRAIN"
    cmd_create = create_command(station=cities,
                                name=f"{station}RETRAIN",
                                length=length,
                                cities_crop=station)
    cmd_retrain = train_command(name=exp_name,
                                pattern=pattern,
                                seq_len=seq_len,
                                batch_size=batch_size,
                                d_model=d_model,
                                d_ff=d_ff,
                                heads=heads,
                                layers=nlayers,
                                val_ratio=0.1,
                                epochs=200,
                                other_args=[f"--load_model {path}/checkpoints/model_best.pth"]
                                )
    cmd_eval = eval_command(name=exp_name,
                            train_pattern=pattern,
                            seq_len=seq_len,
                            d_model=d_model,
                            d_ff=d_ff,
                            heads=heads,
                            layers=nlayers,
                            batch_size=batch_size,
                            )

    with open(job_name, "w") as fh:
        fh.write(start)
        fh.write(cmd_create)
        fh.write(cmd_retrain)
        fh.write(cmd_eval)


if __name__ == "__main__":
    for c in cities:
        exp_name = f"{c}_retrain"
        job_name = f"slurm/{c}_270921.slurm"
        length = 300
        make_retrain(exp_name=exp_name,
                     # path=f"experiments/single_station/240921/big/gpu/{c}_extrapolation/",
                     path=f"experiments/{c}_extrapolation/",
                     station=c,
                     length=length,
                     job_name=job_name)

        os.system(f"sbatch {job_name}")
