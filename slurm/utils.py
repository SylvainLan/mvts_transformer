import os


def make_slurm(job_name,
               data_input=None,
               seq_len=10,
               epochs=1000,
               batch_size=32,
               d_model=16,
               d_ff=128,
               layers=2,
               heads=4,
               load_model=None,
               test_pattern=None,
               launch=False,
               mode="regression",
               data_input_rt=None):
    start_script = ("#!/bin/bash\n" +
                    f"#SBATCH --job-name={mode}\n" +
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


    data_dir_eval = "data/regression/HUR/"
    if mode == "regression":
        data_dir = "data/regression/HUR/"
    elif mode == "imputation":
        data_dir = "data/regression/ImputationHUR/"
        data_dir_retrain = "data/regression/HUR/"

    pattern = data_input.split("_")[-1].split(".")[0]
    if data_input_rt is not None:
        pattern_rt = data_input_rt.split("_")[-1].split(".")[0]

    name = f"{mode.upper()}_{pattern}_{d_model}_{d_ff}_{heads}_{layers}_{seq_len}"

    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {name} " +\
              f"--task {mode} " +\
              f"--data_dir {data_dir} " +\
              "--data_class hur " +\
              f"--pattern {pattern} " +\
              "--val_ratio 0.2 " +\
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
              "--seed 1\n"

    if mode == "imputation":
        name_rt = f"FINETUNE_{pattern}_{d_model}_{d_ff}_{heads}_{layers}_{seq_len}"
        command_rt = "python src/main.py " +\
                     "--output_dir experiments " +\
                     f"--name {name_rt} " +\
                     "--task regression " +\
                     f"--data_dir {data_dir_retrain} " +\
                     "--data_class hur " +\
                     f"--pattern {pattern_rt} " +\
                     "--val_ratio 0.2 " +\
                     f"--epochs {epochs} " +\
                     "--optimizer RAdam " +\
                     f"--batch_size {batch_size} " +\
                     f"--d_model {d_model} " +\
                     f"--dim_feedforward {d_ff} " +\
                     f"--num_heads {heads} " +\
                     f"--num_layers {layers} " +\
                     f"--max_seq_len {seq_len} " +\
                     f"--load_model experiments/{name}/checkpoints/model_best.pth " +\
                     "--no_timestamp " +\
                     "--seed 1 " +\
                     "--change_output\n"
        command = command + command_rt
        name = name_rt


    command_2 = "python src/eval.py " +\
                "--output_dir experiments " +\
                f"--name {name} " +\
                "--task regression " +\
                f"--data_dir {data_dir_eval} " +\
                "--data_class hur " +\
                f"--pattern {pattern} " +\
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

    command_3 = f"cp {job_name} experiments/{name}/\n" +\
                f"rm {job_name}"

    with open(job_name, 'w') as fh:
        fh.write(start_script)
        fh.write(command)
        fh.write(command_2)
        fh.write(command_3)

    if launch:
        os.system(f"sbatch {job_name}")
