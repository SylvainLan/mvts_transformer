import os

def start_script(job_name, mode="cpu"):
    if mode == "cpu":
        partition = "cpu_long"
        # cpu_per_task = 4
        cpu_per_task = 2
        conda_load = ""
        gpu_load = ""
    elif mode == "gpu":
        partition = "gpu"
        cpu_per_task = 2
        conda_load = "module load cuda/10.2.89/intel-19.0.3.199\n"
        gpu_load = "#SBATCH --gres=gpu:1\n"
    else:
        raise AttributeError("soit gpu soit cpu")

    command = ("#!/bin/bash\n" +
                f"#SBATCH --job-name={job_name}\n" +
                "#SBATCH --output=output_ruche/%x.o%j\n" +
                "#SBATCH --time=24:00:00\n" +
                "#SBATCH --ntasks=1\n" +
                f"#SBATCH --cpus-per-task={cpu_per_task}\n"
                "#SBATCH --mem=64GB\n" +
                f"#SBATCH --partition={partition}\n" +
                gpu_load +
                "\n" +
                "module purge\n" +
                "module load anaconda3/2020.02/gcc-9.2.0\n"+
                conda_load +
                "source activate pytorch_vnew\n" +
                "cd ${SLURM_SUBMIT_DIR}\n\n")
    return command


def create_command(station, name, n_splits=1, station_val=None):
    if not hasattr(station, "__len__"):
        station = [station]
    station = [str(s) for s in station]
    str_station = " ".join(station)
    if station_val is not None:
        if not hasattr(station_val, "__len__"):
            station_val = [station_val]
        station_val = [str(s) for s in station_val]
        str_station_val = "--cities_val " + " ".join(station_val)
    else:
        str_station_val = ""

    command = f"python src/create_data.py --cities_train {str_station} --name {name} --n_splits {n_splits} {str_station_val}\n"
    return command


def train_command(name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None, val_pattern=None):
    if val_pattern is not None:
        val_line = f"--val_pattern {val_pattern} "
    else:
        val_line = f"--val_ratio {val_ratio} "
    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              f"--task regression " +\
              f"--data_dir data/regression/HUR/ " +\
              "--data_class hur " +\
              f"--pattern {pattern} " +\
              f"{val_line}" +\
              f"--epochs {epochs} " +\
              "--optimizer RAdam " +\
              f"--batch_size {batch_size} " +\
              f"--d_model {d_model} " +\
              f"--dim_feedforward {d_ff} " +\
              f"--num_heads {heads} " +\
              f"--num_layers {layers} " +\
              f"--max_seq_len {seq_len} " +\
              "--no_timestamp " +\
              "--normalization_layer LayerNorm " +\
              "--seed 1 "
    if other_args is not None:
        for arg in other_args:
            command += arg + " "
    command += "\n"

    return command


def eval_command(name, train_pattern, seq_len, eval_pattern="FULL", batch_size=32, d_model=16, d_ff=128, heads=4, layers=2):
    command = "python src/eval.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              "--task regression " +\
              f"--data_dir data/regression/HUR/ " +\
              "--data_class hur " +\
              f"--pattern {train_pattern} " +\
              f"--load_model experiments/{name}/checkpoints/model_best.pth " +\
              "--test_pattern FULL " +\
              f"--batch_size {batch_size} " +\
              f"--d_model {d_model} " +\
              f"--dim_feedforward {d_ff} " +\
              f"--num_heads {heads} " +\
              f"--num_layers {layers} " +\
              f"--max_seq_len {seq_len} " +\
              "--normalization_layer LayerNorm " +\
              "--no_timestamp\n"
    return command


def clean_command(job_name_long, exp_name, other_args=None):
    # TODO attention à l'etoile, peut être que ça peut casser des trucs
    command = f"cp {job_name_long} experiments/{exp_name}*/\n" +\
              f"rm {job_name_long}\n"
    if other_args is not None:
        for arg in other_args:
            command += arg + "\n"
    return command


def imputation_command(name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None):
    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              f"--task imputation " +\
              f"--data_dir data/regression/ImputationHUR/ " +\
              "--data_class hur " +\
              f"--pattern {pattern} " +\
              f"--val_ratio {val_ratio} " +\
              f"--epochs {epochs} " +\
              "--optimizer RAdam " +\
              f"--batch_size {batch_size} " +\
              f"--d_model {d_model} " +\
              f"--dim_feedforward {d_ff} " +\
              f"--num_heads {heads} " +\
              f"--num_layers {layers} " +\
              f"--max_seq_len {seq_len} " +\
              "--no_timestamp " +\
              "--normalization_layer LayerNorm " +\
              "--seed 1 "

    if other_args is not None:
        for arg in other_args:
            command += arg + " "
    command += "\n"

    return command


def retrain_command(name, imputation_name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None):
    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              "--task regression " +\
              f"--data_dir data/regression/HUR/ " +\
              "--data_class hur " +\
              f"--pattern {pattern} " +\
              f"--val_ratio {val_ratio} " +\
              f"--epochs {epochs} " +\
              "--optimizer RAdam " +\
              f"--batch_size {batch_size} " +\
              f"--d_model {d_model} " +\
              f"--dim_feedforward {d_ff} " +\
              f"--num_heads {heads} " +\
              f"--num_layers {layers} " +\
              f"--max_seq_len {seq_len} " +\
              f"--load_model experiments/{imputation_name}/checkpoints/model_best.pth " +\
              "--no_timestamp " +\
              "--seed 1 " +\
              "--change_output "

    if other_args is not None:
        for arg in other_args:
            command += arg + " "
    command += "\n"

    return command
