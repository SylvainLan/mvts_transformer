import itertools
import subprocess
import argparse


nlayers = [2, 4]
heads = [4, 8]
d_model = [32, 64]
seq_len = [30, 60]
d_ff = [64, 128]
batch_size = [32, 64]


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ruche")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse()
    mode = args.mode

    for i, (n, h, d, s, ff, b) in enumerate(itertools.product(nlayers,
                                                             heads,
                                                             d_model,
                                                             seq_len,
                                                             d_ff,
                                                             batch_size)):
        if mode == "ruche":
            subprocess.call(["python", "slurm/extrapolation.py",
                             "--nlayers", f"{n}",
                             "--heads", f"{h}",
                             "--d_model", f"{d}",
                             "--seq_len", f"{s}",
                             "--d_ff", f"{ff}",
                             "--batch_size", f"{b}"])
        else:
            subprocess.call(["mkdir", f"experiments/single_station/210921/exp{i}"])
            with open("experiments/single_station/210921/README", "a") as f:
                f.write(f"exp{i} : nlayers {n}, heads {h}, d_model {d}, seq_len {s}, d_ff {ff}, batch_size {b}\n")
