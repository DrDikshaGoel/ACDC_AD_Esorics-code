import os
args = {"fn": "r4000", "budget": 5, "start_node_number": 20, "seed": 3}

if "seed" in os.environ:
    assert os.environ["seed"] in "0123456789"
    args["seed"] = int(os.environ["seed"])

if "fn" in os.environ:
    assert os.environ["fn"] in ["r500", "r1000", "r2000", "r4000", "ads5"]
    args["fn"] = os.environ["fn"]

if "story_name" in os.environ:
    assert os.environ["story_name"] in ["reject_diversity", "reject_value", "greedy_defense"]
    args["story_name"] = os.environ["story_name"]

if "distribution" in os.environ:
    assert os.environ["distribution"] in ["I", "P", "N"]
    args["distribution"] = os.environ["distribution"]
    distribution = args["distribution"]
"""
story_name = "reject_diversity"
args["story_name"] = story_name
distribution = "P"
args["distribution"] = distribution
"""

train_env = 20
test_env = 20
pop_size = 20
mutate_p = 0.5
absolute_diff = 0.1
ec_iterations = 20000  # only a few seconds
start_from_best_defense_p = 0
reset_after_epoch = 60 ####50   #40 
stop_after_epoch = 1480
epoch = 1500 ###750###150####750###150##200####50##700  ###200  #700
"""
train_env = 1
test_env = 1
pop_size = 1
mutate_p = 0.5
absolute_diff = 0.1
ec_iterations = 100  # only a few seconds
start_from_best_defense_p = 0
reset_after_epoch = 20 ####50   #40
stop_after_epoch = 35
epoch = 40 ###750###150####750###150##200####50##700  ###200  #700
"""
# FROM HERE
base_prob = 0.5
upper_prob = 0.5
num_snapshots = 50 # number of graph snapshots we are considering
simplify_after_epoch = 15###15#call simplification agent after these epochs
simplify_collect_data_for_episodes = 10000   #10000 ##keep the number high #collect in somplifiaction phase data for these many episodes
simplify_max_edgeto_block=0.1 #0.05
score_difference_threshold = 0.05 #0.02 ##set to low values
NNprun_formin = 2
NNweight_of_ori = 0.99


#step_per_collect = 20
#each_snap_step = 20    #change next snapshot sfter these steps in reset function
preference_threshold = 0.9

#NN_prun_run_after = 300
#NN_prun_run_for = 30
list
