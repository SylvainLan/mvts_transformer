from utils import make_slurm
import itertools

seq_lens = [16]
d_models = [16, 32]
d_ffs = [128]
layers = [2]
heads = [4]

for i, (d_model, d_ff, layer, head, seq_len) in enumerate(itertools.product(d_models, d_ffs, layers, heads, seq_lens)):
    # JOBS for S6
    make_slurm(job_name=f"slurm/reg_S6_010921_{i}.slurm",
               data_input="data/regression/HUR/HUR_S6.csv",
               seq_len=seq_len,
               d_model=d_model,
               d_ff=d_ff,
               layers=layer,
               heads=head,
               launch=True,
               epochs=2000,
               mode="imputation",
               data_input_rt="data/regression/HUR/HUR_S6.csv")

    # # JOBS for full
    # make_slurm_regression(job_name=f"slurm/reg_full_010921_{i}.slurm",
    #                       data_input="data/regression/HUR/HUR_FULL.csv",
    #                       seq_len=seq_len,
    #                       d_model=d_model,
    #                       d_ff=d_ff,
    #                       layers=layer,
    #                       heads=head,
    #                       launch=True,
    #                       epochs=2000)

    # # JOBS for imputation on small
    # make_slurm_imputation(job_name=f"slurm/imput_small_100821_{i}.slurm",
    #                       data_input="data/regression/ImputationHUR/ImputationHUR_SMALL.csv",
    #                       data_input_2="data/regression/HUR/HUR_S6.csv", # CHANGE AFTER
    #                       seq_len=seq_len,
    #                       d_model=d_model,
    #                       d_ff=d_ff,
    #                       layers=layer,
    #                       heads=head,
    #                       launch=True,
    #                       epochs=2000)
    # # JOBS for imputation on medium
    # make_slurm_imputation(job_name=f"slurm/imput_medium_100821_{i}.slurm",
    #                       data_input="data/regression/ImputationHUR/ImputationHUR_MEDIUM.csv",
    #                       data_input_2="data/regression/HUR/HUR_S6.csv", # CHANGE AFTER
    #                       seq_len=seq_len,
    #                       d_model=d_model,
    #                       d_ff=d_ff,
    #                       layers=layer,
    #                       heads=head,
    #                       launch=True,
    #                       epochs=2000)
