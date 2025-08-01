#!/bin/bash

# Activate the Conda environment
source activate cs285


# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" 

# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions False --cql_actions_weight 0.0 --temp 0 --detach True --num_action_samples 0
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.001 --temp 0.7 --detach True --num_action_samples 10
python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0 --temp 1 --detach False --num_action_samples 10 --step_count 25000
python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.001 --temp 1 --detach False --num_action_samples 10 --step_count 25000
python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 1 --detach False --num_action_samples 10 --step_count 25000
python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.1 --temp 1 --detach False --num_action_samples 10 --step_count 25000
python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 1 --temp 1 --detach True --num_action_samples 10 --step_count 25000
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 2 --temp 1 --detach True --num_action_samples 10 --step_count 25000
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 1 --temp 1 --detach True --num_action_samples 10 --step_count 25000

# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.001 --temp 0.7 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 0.7 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.1 --temp 0.7 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 1 --temp 0.7 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 5 --temp 0.7 --detach False --num_action_samples 10

# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions False --cql_actions_weight 1 --temp 1 --detach True --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 1 --detach True --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 0.7 --detach True --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 0.6 --detach True --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 0.01 --temp 0.5 --detach True --num_action_samples 10

# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions False --cql_actions_weight 1 --temp 1 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 1 --temp 0.7 --detach False --num_action_samples 10
# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/Train_ccbf_withlogging.py" --use_cql_actions True --cql_actions_weight 1 --temp 0.5 --detach False --num_action_samples 10







# chmod +x "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/trainer.sh"
#caffeinate -i "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF/scripts/trainer.sh"

# python "/Users/i.k.tabbara/Documents/python directory/simulator_custom_learn_CBF_CCBF_images/scripts/train_CBF_with VAEDYNAMICS_without_sequences.py" --use_cql_actions True --cql_actions_weight 1 --temp 0.65 --detach True --batch_size 256 --safe_distance 7