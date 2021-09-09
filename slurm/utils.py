import os

def start_script(job_name):
    command = ("#!/bin/bash\n" +
                f"#SBATCH --job-name={job_name}\n" +
                "#SBATCH --output=output_ruche/%x.o%j\n" +
                "#SBATCH --time=24:00:00\n" +
                "#SBATCH --ntasks=1\n" +
                "#SBATCH --cpus-per-task=4\n"
                "#SBATCH --mem=64GB\n" +
                # "#SBATCH --gres=gpu:1\n" +
                # "#SBATCH --partition=gpu\n" +
                "#SBATCH --partition=cpu_long\n" +
                "\n" +
                "module purge\n" +
                "module load anaconda3/2020.02/gcc-9.2.0\n"+
                # "module load cuda/10.2.89/intel-19.0.3.199\n"+
                "source activate pytorch_vnew\n" +
                "cd ${SLURM_SUBMIT_DIR}\n\n")
    return command


def train_command(name, pattern, seq_len, val_ratio=0.2, epochs=2000, batch_size=32, d_model=16, d_ff=128, heads=4, layers=2, other_args=None):
    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              f"--task regression " +\
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


def clean_command(job_name_long, exp_name):
    command = f"cp {job_name_long} experiments/{exp_name}/\n" +\
              f"rm {job_name_long}\n"
    return command


#if mode == "imputation":
#    name_rt = f"FINETUNE_{pattern}_{d_model}_{d_ff}_{heads}_{layers}_{seq_len}"
#    command_rt = "python src/main.py " +\
#                 "--output_dir experiments " +\
#                 f"--name {name_rt} " +\
#                 "--task regression " +\
#                 f"--data_dir {data_dir_retrain} " +\
#                 "--data_class hur " +\
#                 f"--pattern {pattern_rt} " +\
#                 "--val_ratio 0.2 " +\
#                 f"--epochs {epochs} " +\
#                 "--optimizer RAdam " +\
#                 f"--batch_size {batch_size} " +\
#                 f"--d_model {d_model} " +\
#                 f"--dim_feedforward {d_ff} " +\
#                 f"--num_heads {heads} " +\
#                 f"--num_layers {layers} " +\
#                 f"--max_seq_len {seq_len} " +\
#                 f"--load_model experiments/{name}/checkpoints/model_best.pth " +\
#                 "--no_timestamp " +\
#                 "--seed 1 " +\
#                 "--change_output\n"
#    command = command + command_rt
#    name = name_rt


#    with open(job_name, 'w') as fh:
#        fh.write(start_script)
#        fh.write(command)
#        fh.write(command_2)
#        fh.write(command_3)

    if launch:
        os.system(f"sbatch {job_name}")
