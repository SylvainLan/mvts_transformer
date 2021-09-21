import itertools
import subprocess
import argparse
import glob


nlayers = [2]
heads = [4]
d_model = [32]
seq_len = [30]
d_ff = [32, 64, 128]
batch_size = [32]


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ruche")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--i_max", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = _parse()
    mode = args.mode
    clean = args.clean
    i_max = args.i_max

    if clean:
        print("cleaning")
        for f in glob.glob("experiments/single_station/210921/exp*"):
            subprocess.call(["rm", "-r", "-f", f])
        subprocess.call(["rm", "experiments/single_station/210921/README"])

    if i_max == 0 and len(glob.glob("experiments/single_station/210921/exp*")) > 1:
            i_max = max([int(f.split("exp")[-1]) for f in glob.glob("experiments/single_station/210921/exp*")]) + 1

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
                             "--batch_size", f"{b}",
                             "--exp_prefix", f"exp{i + i_max}"])
        else:
            subprocess.call(["mkdir", f"experiments/single_station/210921/exp{i + i_max}"])
            with open("experiments/single_station/210921/README", "a") as f:
                f.write(f"exp{i + i_max} : nlayers {n}, heads {h}, d_model {d}, seq_len {s}, d_ff {ff}, batch_size {b}\n")
