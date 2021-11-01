import os

def start_script(job_name, mode="cpu", ncpus=2, ngrid=0):
    if "gpu" in mode:
        partition = "gpu"
        time = "12"
        cpu_per_task = 2
        conda_load = "module load cuda/10.2.89/intel-19.0.3.199\n"
        gpu_load = "#SBATCH --gres=gpu:1\n"
    elif mode == "cpu_long" or mode == "cpu_med":
        time = {"cpu_long": "24", "cpu_med": "04"}[mode]
        partition = mode
        cpu_per_task = ncpus
        conda_load = ""
        gpu_load = ""
    else:
        raise AttributeError("soit gpu soit cpu_med soit cpu_long")

    if ngrid > 0:
         grid = f"#SBATCH --array=0-{ngrid - 1}\n"
    else:
        grid = ""

    command = ("#!/bin/bash\n" +
                f"#SBATCH --job-name={job_name}\n" +
                "#SBATCH --output=output_ruche/%x.o%j\n" +
                f"#SBATCH --time={time}:00:00\n" +
                "#SBATCH --ntasks=1\n" +
                f"#SBATCH --cpus-per-task={cpu_per_task}\n"
                "#SBATCH --mem=64GB\n" +
                f"#SBATCH --partition={partition}\n" +
                grid +
                gpu_load +
                "\n" +
                "module purge\n" +
                "module load anaconda3/2020.02/gcc-9.2.0\n"+
                conda_load +
                "source activate pytorch_vnew\n" +
                "cd ${SLURM_SUBMIT_DIR}\n\n")
    return command


def create_command(station, name, n_splits=1, station_val=None, length=None, cities_crop=None):
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
    if length is None:
        str_length = ""
    else:
        str_length = f"--length {length}"
    if cities_crop is None:
        str_cities_crop = ""
    else:
        if not hasattr(cities_crop, "__len__"):
            cities_crop = [cities_crop]
        cities_crop = [str(c) for c in cities_crop]
        str_cities_crop = f"--cities_crop " + " ".join(cities_crop)

    command = f"python src/create_data.py --cities_train {str_station} --name {name} --n_splits {n_splits} {str_station_val} {str_length} {str_cities_crop}\n"
    return command


def train_command(name, pattern, seq_len, batch_size, d_model, d_ff, heads, layers, val_ratio=0.2, epochs=2000, other_args=None, val_pattern=None, stdout=True):
    if val_pattern is not None:
        val_line = f"--val_pattern {val_pattern} "
        val_line = ["--val_pattern", f"{val_pattern}"]
    else:
        val_line = f"--val_ratio {val_ratio} "
        val_line = ["--val_ratio", f"{val_ratio}"]
    command = ["python", "src/main.py",
               "--output_dir", "experiments",
               "--name", f"{name}",
               "--task", "regression",
               "--data_dir", "data/regression/HUR",
               "--data_class", "hur",
               "--pattern", f"{pattern}",
               *val_line,
               "--epochs", f"{epochs}",
               "--optimizer", "RAdam",
               "--batch_size", f"{batch_size}",
               "--d_model", f"{d_model}",
               "--dim_feedforward", f"{d_ff}",
               "--num_heads", f"{heads}",
               "--num_layers", f"{layers}",
               "--max_seq_len", f"{seq_len}",
               "--no_timestamp",
               "--normalization_layer", "LayerNorm",
               "--seed", "1"
               ]
    #command = "python src/main.py " +\
    #          "--output_dir experiments " +\
    #          f"--name {name} " +\
    #          f"--task regression " +\
    #          f"--data_dir data/regression/HUR/ " +\
    #          "--data_class hur " +\
    #          f"--pattern {pattern} " +\
    #          f"{val_line}" +\
    #          f"--epochs {epochs} " +\
    #          "--optimizer RAdam " +\
    #          f"--batch_size {batch_size} " +\
    #          f"--d_model {d_model} " +\
    #          f"--dim_feedforward {d_ff} " +\
    #          f"--num_heads {heads} " +\
    #          f"--num_layers {layers} " +\
    #          f"--max_seq_len {seq_len} " +\
    #          "--no_timestamp " +\
    #          "--normalization_layer LayerNorm " +\
    #          "--seed 1 "
    if other_args is not None:
        for arg in other_args:
            #command += arg + " "
            command.append(arg)
    #command += "\n"
    #return command
    if stdout:
        return " ".join(command) + "\n"
    else:
        return command



def eval_command(name, train_pattern, seq_len, batch_size, d_model, d_ff, heads, layers, eval_pattern="FULL", stdout=True):
    command = ["python", "src/eval.py" ,
              "--output_dir", "experiments",
              f"--name", f"{name}",
              "--task", "regression",
              f"--data_dir", "data/regression/HUR/",
              "--data_class", "hur",
              f"--pattern", f"{train_pattern}",
              f"--load_model", f"experiments/{name}/checkpoints/model_best.pth",
              "--test_pattern", "FULL",
              f"--batch_size", f"{batch_size}",
              f"--d_model", f"{d_model}",
              f"--dim_feedforward", f"{d_ff}",
              f"--num_heads", f"{heads}",
              f"--num_layers", f"{layers}",
              f"--max_seq_len", f"{seq_len}",
              "--normalization_layer",  "LayerNorm",
              "--no_timestamp"]

    #command = "python src/eval.py " +\
    #          "--output_dir experiments " +\
    #          f"--name {name} " +\
    #          "--task regression " +\
    #          f"--data_dir data/regression/HUR/ " +\
    #          "--data_class hur " +\
    #          f"--pattern {train_pattern} " +\
    #          f"--load_model experiments/{name}/checkpoints/model_best.pth " +\
    #          "--test_pattern FULL " +\
    #          f"--batch_size {batch_size} " +\
    #          f"--d_model {d_model} " +\
    #          f"--dim_feedforward {d_ff} " +\
    #          f"--num_heads {heads} " +\
    #          f"--num_layers {layers} " +\
    #          f"--max_seq_len {seq_len} " +\
    #          "--normalization_layer LayerNorm " +\
    #          "--no_timestamp\n"
    #return command
    if stdout:
        return " ".join(command) + "\n"
    else:
        return command


def clean_command(job_name_long, exp_name, other_args=None):
    # TODO attention à l'etoile, peut être que ça peut casser des trucs
    command = f"cp {job_name_long} experiments/{exp_name}/\n" +\
              f"rm {job_name_long}\n"
    if other_args is not None:
        for arg in other_args:
            command += arg + "\n"
    return command


# def imputation_command(name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None):
#     command = "python src/main.py " +\
#               "--output_dir experiments " +\
#               f"--name {name} " +\
#               f"--task imputation " +\
#               f"--data_dir data/regression/ImputationHUR/ " +\
#               "--data_class hur " +\
#               f"--pattern {pattern} " +\
#               f"--val_ratio {val_ratio} " +\
#               f"--epochs {epochs} " +\
#               "--optimizer RAdam " +\
#               f"--batch_size {batch_size} " +\
#               f"--d_model {d_model} " +\
#               f"--dim_feedforward {d_ff} " +\
#               f"--num_heads {heads} " +\
#               f"--num_layers {layers} " +\
#               f"--max_seq_len {seq_len} " +\
#               "--no_timestamp " +\
#               "--normalization_layer LayerNorm " +\
#               "--seed 1 "
# 
#     if other_args is not None:
#         for arg in other_args:
#             command += arg + " "
#     command += "\n"
# 
#     return command


# def retrain_command(name, imputation_name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None):
#     command = "python src/main.py " +\
#               "--output_dir experiments " +\
#               f"--name {name} " +\
#               "--task regression " +\
#               f"--data_dir data/regression/HUR/ " +\
#               "--data_class hur " +\
#               f"--pattern {pattern} " +\
#               f"--val_ratio {val_ratio} " +\
#               f"--epochs {epochs} " +\
#               "--optimizer RAdam " +\
#               f"--batch_size {batch_size} " +\
#               f"--d_model {d_model} " +\
#               f"--dim_feedforward {d_ff} " +\
#               f"--num_heads {heads} " +\
#               f"--num_layers {layers} " +\
#               f"--max_seq_len {seq_len} " +\
#               f"--load_model experiments/{imputation_name}/checkpoints/model_best.pth " +\
#               "--no_timestamp " +\
#               "--seed 1 " +\
#               "--change_output "
# 
#     if other_args is not None:
#         for arg in other_args:
#             command += arg + " "
#     command += "\n"
# 
#     return command
