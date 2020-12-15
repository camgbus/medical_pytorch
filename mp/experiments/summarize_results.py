# ------------------------------------------------------------------------------
# Summarizing results by extracting the last and best (acc. to val data if 
# present, otherwise train) state to write in the .
# 
# In the experiment review.json file, the mean and std of these results are 
# reported.
# ------------------------------------------------------------------------------

from mp.utils.load_restore import pkl_load



path = "/home/cam/Documents/server_results/BaseCardiac_2d_bigLR/0/results/results"

exp_results = pkl_load(path)

results = exp_results[0]

summary_metric = 'Mean_ScoreDice[left ventricle]'

# Dataset name for getting the best epoch
best_epoch_data = 'MM_Challenge[Vendor:A][Labels:(0, 1, 0, 0)]_train'

# Get max and best epochs
max_e = results.get_max_epoch(summary_metric)
best_e = results.get_best_epoch(summary_metric, data=best_epoch_data)

# Get results for all datasets for the the max and best epoch
data_results_max = results.get_epoch_results(summary_metric, max_e)
data_results_best = results.get_epoch_results(summary_metric, best_e)
