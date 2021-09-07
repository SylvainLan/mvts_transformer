import os
import string


def make_slurm(job_name,
               city_test=6,
               city_val=None,
               seq_len=10,
               epochs=1000,
               batch_size=32,
               d_model=16,
               d_ff=128,
               layers=2,
               heads=4,
               launch=False):
    job_name = f"slurm/{city_val}_train_val.slurm"
    start_script = ("#!/bin/bash\n" +
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


    data_dir = "data/regression/HUR/"


    pattern = f"{city_val}_{city_test}"
    command_create = f"python src/create_data.py --cities_test {city_test} --cities_val {city_val} --name {pattern}\n"
    exp_name = f"{city_val}_eval_{city_test}_test"

    command = "python src/main.py " +\
              "--output_dir experiments " +\
              f"--name {exp_name} " +\
              f"--task regression " +\
              f"--data_dir {data_dir} " +\
              "--data_class hur " +\
              f"--pattern {pattern}TRAIN " +\
              f"--val_pattern {pattern}VAL " +\
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

    command_2 = "python src/eval.py " +\
                "--output_dir experiments " +\
                f"--name {exp_name} " +\
                "--task regression " +\
                f"--data_dir {data_dir} " +\
                "--data_class hur " +\
                f"--pattern {pattern}TRAIN " +\
                f"--load_model experiments/{exp_name}/checkpoints/model_best.pth " +\
                f"--test_pattern FULL " +\
                f"--batch_size {batch_size} " +\
                f"--d_model {d_model} " +\
                f"--dim_feedforward {d_ff} " +\
                f"--num_heads {heads} " +\
                f"--num_layers {layers} " +\
                f"--max_seq_len {seq_len} " +\
                "--normalization_layer LayerNorm " +\
                "--no_timestamp\n"

    command_3 = f"cp {job_name} experiments/{exp_name}/\n" +\
                f"rm {job_name}\n" +\
                f"rm data/regression/HUR/HUR_{pattern}*.csv\n"

    with open(job_name, 'w') as fh:
        fh.write(start_script)
        fh.write(command_create)
        fh.write(command)
        fh.write(command_2)
        fh.write(command_3)

    if launch:
        os.system(f"sbatch {job_name}")

cities = [15, 19, 27, 34, 50, 54, 77, 78, 84, 99]
city_test = 6

if __name__ == "__main__":
    for city in cities:
        make_slurm(job_name=f"slurm/trainval_{city}.slurm", city_test=city_test, city_val=city, seq_len=16, epochs=2000, batch_size=32, d_model=16, d_ff=128, layers=2, heads=4, launch=True)
