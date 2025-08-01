import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp

# Silence cvxopt output
solvers.options['show_progress'] = False

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

# Import required modules
from modules.network import MLPGaussianActor, CBF
from modules.dataset import Dataset
from envs.car import DubinsCarEnv

# Define constants
DATASET_PATH = "safe_rl_dataset.npz"
GOAL_POSITION = np.array([20, 21])

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_9999cql_statesFalsecql_actionsFalse_cql_states_weight0.1_cql_actions_weight1e-05_843.pt"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsFalse_cql_states_weight0.1_cql_actions_weight0.01_225.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999cql_statesFalsecql_actionsTrue_cql_states_weight0.1_cql_actions_weight0.01_122.pt" 
##i think above ccbf checkpt was good


# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsFalse_cql_states_weight0.1_cql_actions_weight0.01_225.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.1_197.pt"
# # ABOVE ARE NORMAL TRAINED CBF AND A CQL CBF with detach 
#for above:
'''
Method                              Avg Reward      Avg Cost        Success %       Collision %
BC (All Data)                       85.00           3.71            99.0            20.0
BC-Safe (Safe Data Only)            81.99           0.29            97.0            4.0
CBF-QP (Goal Reaching)              67.89           0.00            75.2            0.0
CBF-QP (BC + CBF)                   79.96           0.00            92.0            0.0
CBF-QP (BC-Safe + CBF)              79.59           0.00            90.8            0.0
CCBF-QP (Goal Reaching)             68.86           0.00            74.6            0.0
CCBF-QP (BC + CBF)                  80.33           0.00            92.4            0.0
CCBF-QP (BC-Safe + CBF)             80.60           0.00            93.4            0.0
'''
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsFalse_cql_states_weight0.1_cql_actions_weight0.01_225.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.1_757.pt"
# ABOVE ARE NORMAL TRAINED CBF AND A CQL CBF without detach
#for above:
'''                 
Method                              Avg Reward      Avg Cost        Success %       Collision %
CBF-QP (Goal Reaching)              67.89           0.00            75.2            0.0            
CBF-QP (BC + CBF)                   79.96           0.00            92.0            0.0            
CBF-QP (BC-Safe + CBF)              79.59           0.00            90.8            0.0            
CCBF-QP (Goal Reaching)             66.29           0.00            71.8            0.0            
CCBF-QP (BC + CBF)                  80.57           0.00            93.0            0.0            
CCBF-QP (BC-Safe + CBF)             78.85           0.00            89.8            0.0    
# '''
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_155.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_742.pt"
# COMPARISON OF ALL METHODS
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              71.03           0.00            80.0            0.0            
# CBF-QP (BC + CBF)                   78.34           0.00            90.0            0.0            
# CBF-QP (BC-Safe + CBF)              82.96           0.00            94.0            0.0            
# CCBF-QP (Goal Reaching)             67.48           0.00            76.0            0.0            
# CCBF-QP (BC + CBF)                  67.52           0.00            72.0            0.0            
# CCBF-QP (BC-Safe + CBF)             74.74           0.00            84.0            0.0   

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_155.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_149pt"
# CBF-QP (Goal Reaching)              71.03           0.00            80.0            0.0            
# CBF-QP (BC + CBF)                   78.34           0.00            90.0            0.0            
# CBF-QP (BC-Safe + CBF)              82.96           0.00            94.0            0.0            
# CCBF-QP (Goal Reaching)             67.48           0.00            76.0            0.0            
# CCBF-QP (BC + CBF)                  67.52           0.00            72.0            0.0            
# CCBF-QP (BC-Safe + CBF)             74.74           0.00            84.0            0.0  


# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_770.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_979.pt"
# COMPARISON OF ALL METHODS
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              72.65           0.00            82.0            0.0            
# CBF-QP (BC + CBF)                   78.33           0.00            90.0            0.0            
# CBF-QP (BC-Safe + CBF)              83.02           0.00            94.0            0.0            
# CCBF-QP (Goal Reaching)             76.32           0.00            88.0            0.0            
# CCBF-QP (BC + CBF)                  75.62           0.00            82.0            0.0            
# CCBF-QP (BC-Safe + CBF)             83.27           0.00            96.0            0.0  


# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_617.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_278.pt"
# ==================================================
# COMPARISON OF ALL METHODS
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              72.81           0.00            82.0            0.0            
# CBF-QP (BC + CBF)                   78.23           0.00            90.0            0.0            
# CBF-QP (BC-Safe + CBF)              82.99           0.00            94.0            0.0            
# CCBF-QP (Goal Reaching)             67.93           0.00            74.0            0.0            
# CCBF-QP (BC + CBF)                  76.87           0.00            84.0            0.0            
# CCBF-QP (BC-Safe + CBF)             82.24           0.00            94.0            0.0   

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_404.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_196.pt"
# CBF-QP (Goal Reaching)              55.95           0.00            58.0            0.0            
# CBF-QP (BC + CBF)                   75.54           0.00            86.0            0.0            
# CBF-QP (BC-Safe + CBF)              78.81           0.00            86.0            0.0            
# CCBF-QP (Goal Reaching)             83.81           18.88           100.0           90.0           
# CCBF-QP (BC + CBF)                  81.57           7.02            92.0            36.0           
# CCBF-QP (BC-Safe + CBF)             84.31           0.20            98.0            4.0  


# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_902.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_856.pt"
# CBF-QP (Goal Reaching)              72.65           0.00            82.0            0.0            
# CBF-QP (BC + CBF)                   78.33           0.00            90.0            0.0            
# CBF-QP (BC-Safe + CBF)              83.02           0.00            94.0            0.0            
# CCBF-QP (Goal Reaching)             59.64           0.00            64.0            0.0            
# CCBF-QP (BC + CBF)                  72.25           0.00            78.0            0.0            
# CCBF-QP (BC-Safe + CBF)             82.22           0.00            94.0            0.0   

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_304.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_335.pt"
# CBF-QP (Goal Reaching)              55.95           0.00            58.0            0.0            
# CBF-QP (BC + CBF)                   75.54           0.00            86.0            0.0            
# CBF-QP (BC-Safe + CBF)              78.81           0.00            86.0            0.0            
# CCBF-QP (Goal Reaching)             59.64           0.00            64.0            0.0            
# CCBF-QP (BC + CBF)                  72.25           0.00            78.0            0.0            
# CCBF-QP (BC-Safe + CBF)             82.22           0.00            94.0            0.0  

CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_502.pt"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_862.pt"
# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              71.14           0.00            80.0            0.0            
# CBF-QP (BC + CBF)                   83.00           0.00            96.0            0.0            
# CBF-QP (BC-Safe + CBF)              80.52           0.00            91.0            0.0            
# CCBF-QP (Goal Reaching)             70.52           0.00            79.0            0.0            
# CCBF-QP (BC + CBF)                  80.22           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             78.93           0.00            90.0            0.0  
# ==================================================
# COMPARISON OF ALL METHODS 300 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.21           3.86            95.7            20.0           
# BC-Safe (Safe Data Only)            82.33           0.58            95.7            5.3            
# CBF-QP (Goal Reaching)              70.28           0.00            77.0            0.0            
# CBF-QP (BC + CBF)                   81.58           0.00            94.7            0.0            
# CBF-QP (BC-Safe + CBF)              81.94           0.00            95.3            0.0            
# CCBF-QP (Goal Reaching)             68.49           0.00            73.0            0.0            
# CCBF-QP (BC + CBF)                  81.13           0.00            93.0            0.0            
# CCBF-QP (BC-Safe + CBF)             80.61           0.00            92.7            0.0            
# (osrl) (base) i.k.tabbara@1J000AL-FYYVJW7 python directory % 
CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_893.pt"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_409.pt"
# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              70.53           0.00            78.0            0.0            
# CBF-QP (BC + CBF)                   82.36           0.00            95.0            0.0            
# CBF-QP (BC-Safe + CBF)              78.89           0.00            88.0            0.0            
# CCBF-QP (Goal Reaching)             70.52           0.00            79.0            0.0            
# CCBF-QP (BC + CBF)                  80.22           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             78.93           0.00            90.0            0.0    
# COMPARISON OF ALL METHODS 300 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.21           3.86            95.7            20.0           
# BC-Safe (Safe Data Only)            82.33           0.58            95.7            5.3            
# CBF-QP (Goal Reaching)              68.48           0.00            73.0            0.0            
# CBF-QP (BC + CBF)                   80.63           0.00            93.0            0.0            
# CBF-QP (BC-Safe + CBF)              81.38           0.00            94.3            0.0            
# CCBF-QP (Goal Reaching)             68.49           0.00            73.0            0.0            
# CCBF-QP (BC + CBF)                  81.13           0.00            93.0            0.0            
# CCBF-QP (BC-Safe + CBF)             80.61           0.00            92.7            0.0          

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_349.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_712.pt"  
# # COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              74.51           0.00            84.0            0.0            
# CBF-QP (BC + CBF)                   83.01           0.00            96.0            0.0            
# CBF-QP (BC-Safe + CBF)              81.08           0.00            92.0            0.0            
# CCBF-QP (Goal Reaching)             77.47           0.00            89.0            0.0            
# CCBF-QP (BC + CBF)                  80.35           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.50           0.00            91.0            0.0  
# ==================================================
# COMPARISON OF ALL METHODS 300 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.21           3.86            95.7            20.0           
# BC-Safe (Safe Data Only)            82.33           0.58            95.7            5.3            
# CBF-QP (Goal Reaching)              74.09           0.00            82.0            0.0            
# CBF-QP (BC + CBF)                   81.42           0.00            94.3            0.0            
# CBF-QP (BC-Safe + CBF)              81.94           0.00            95.3            0.0            
# CCBF-QP (Goal Reaching)             76.80           0.00            85.7            0.0            
# CCBF-QP (BC + CBF)                  81.69           0.00            94.0            0.0            
# CCBF-QP (BC-Safe + CBF)             80.82           0.00            93.0            0.0      

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_557.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_334.pt"  
# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              76.00           0.00            86.0            0.0            
# CBF-QP (BC + CBF)                   83.00           0.00            96.0            0.0            
# CBF-QP (BC-Safe + CBF)              79.97           0.00            90.0            0.0            
# CCBF-QP (Goal Reaching)             77.47           0.00            89.0            0.0            
# CCBF-QP (BC + CBF)                  80.35           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.50           0.00            91.0            0.0     
# ==================================================
# COMPARISON OF ALL METHODS 300 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    COMPAREABLE TO CBF 349
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.21           3.86            95.7            20.0           
# BC-Safe (Safe Data Only)            82.33           0.58            95.7            5.3            
# CBF-QP (Goal Reaching)              75.40           0.00            83.3            0.0            
# CBF-QP (BC + CBF)                   81.41           0.00            94.3            0.0            
# CBF-QP (BC-Safe + CBF)              81.76           0.00            95.0            0.0            
# CCBF-QP (Goal Reaching)             76.80           0.00            85.7            0.0            
# CCBF-QP (BC + CBF)                  81.69           0.00            94.0            0.0            
# CCBF-QP (BC-Safe + CBF)             80.82           0.00            93.0            0.0    

#special runs below size 128 and 10000 runs
CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt"
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_563.pt" 
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              77.65           0.00            89.0            0.0            
# CBF-QP (BC + CBF)                   82.96           0.00            96.0            0.0    ##THIS (159) CAN BE GOOD 159 VS468 ARE SAME AND WE WILL USE THEIR FIGURES    
# CBF-QP (BC-Safe + CBF)              80.54           0.00            91.0            0.0            
# CCBF-QP (Goal Reaching)             78.87           0.00            92.0            0.0            
# CCBF-QP (BC + CBF)                  80.46           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.52           0.00            91.0            0.0 
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_563.pt"  
# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              77.58           0.00            89.0            0.0            
# CBF-QP (BC + CBF)                   83.04           0.00            96.0            0.0            
# CBF-QP (BC-Safe + CBF)              81.13           0.00            92.0            0.0            
# CCBF-QP (Goal Reaching)             78.87           0.00            92.0            0.0            
# CCBF-QP (BC + CBF)                  80.46           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.52           0.00            91.0            0.0  
# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt"
# ==================================================
# COMPARISON OF ALL METHODS 400 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.81           3.95            96.5            20.8           
# BC-Safe (Safe Data Only)            80.74           0.74            93.0            7.2            
# CBF-QP (Goal Reaching)              78.17           0.00            88.8            0.0            
# CBF-QP (BC + CBF)                   82.97           0.00            97.0            0.0            
# CBF-QP (BC-Safe + CBF)              81.76           0.00            94.5            0.0            
# CCBF-QP (Goal Reaching)             77.95           0.00            88.2            0.0            
# CCBF-QP (BC + CBF)                  82.31           0.00            95.8            0.0    

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_19999_182.pt"


# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"##idbf
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              18.62           122.84          0.0             100.0          
# CBF-QP (BC + CBF)                   20.49           0.47            2.0             2.0            
# CBF-QP (BC-Safe + CBF)              22.13           0.00            4.0             0.0            
# CCBF-QP (Goal Reaching)             78.59           0.00            91.0            0.0            
# CCBF-QP (BC + CBF)                  80.46           0.00            92.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.49           0.00            91.0            0.0   

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt" ##pure cbf
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"  ##pure ccbf
# ==================================================
# COMPARISON OF ALL METHODS 300 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.21           3.86            95.7            20.0           
# BC-Safe (Safe Data Only)            82.33           0.58            95.7            5.3            
# CBF-QP (Goal Reaching)              78.13           0.00            88.7            0.0            
# CBF-QP (BC + CBF)                   81.38           0.00            94.3            0.0            
# CBF-QP (BC-Safe + CBF)              82.08           0.00            95.7            0.0            
# CCBF-QP (Goal Reaching)             78.21           0.00            88.3            0.0            
# CCBF-QP (BC + CBF)                  81.94           0.00            94.3            0.0            
# CCBF-QP (BC-Safe + CBF)             81.02           0.00            93.3            0.0      



# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt" ##pure cbf
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"  ##pure ccbf
#seed 42
# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              77.65           0.00            89.0            0.0            
# CBF-QP (BC + CBF)                   82.96           0.00            96.0            0.0            
# CBF-QP (BC-Safe + CBF)              80.54           0.00            91.0            0.0            
# CCBF-QP (Goal Reaching)             79.58           0.00            93.0            0.0            
# CCBF-QP (BC + CBF)                  81.00           0.00            93.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.54           0.00            91.0            0.0   

# CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt"

# ==================================================
# COMPARISON OF ALL METHODS 100 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       84.02           2.98            98.0            15.0           
# BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# CBF-QP (Goal Reaching)              18.62           122.84          0.0             100.0          
# CBF-QP (BC + CBF)                   20.49           0.47            2.0             2.0            
# CBF-QP (BC-Safe + CBF)              22.13           0.00            4.0             0.0            
# CCBF-QP (Goal Reaching)             79.58           0.00            93.0            0.0            
# CCBF-QP (BC + CBF)                  81.00           0.00            93.0            0.0            
# CCBF-QP (BC-Safe + CBF)             79.54           0.00            91.0            0.0     







CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"#cbf
# CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"#ccbf
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"##idbf
CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"##idbf
# # COMPARISON OF ALL METHODS 100 runs
# # ==================================================
# # Method                              Avg Reward      Avg Cost        Success %       Collision %    
# # -----------------------------------------------------------------------------------------------
# # BC (All Data)                       84.02           2.98            98.0            15.0           
# # BC-Safe (Safe Data Only)            80.99           0.53            94.0            6.0            
# # CBF-QP (Goal Reaching)              77.36           0.00            89.0            0.0            
# # CBF-QP (BC + CBF)                   83.04           0.00            96.0            0.0            
# # CBF-QP (BC-Safe + CBF)              80.61           0.00            91.0            0.0            
# # CCBF-QP (Goal Reaching)             78.59           0.00            91.0            0.0            
# # CCBF-QP (BC + CBF)                  80.46           0.00            92.0            0.0            
# # CCBF-QP (BC-Safe + CBF)             79.49           0.00            91.0            0.0   

# ==================================================
# COMPARISON OF ALL METHODS 500 runs
# ==================================================
# Method                              Avg Reward      Avg Cost        Success %       Collision %    
# -----------------------------------------------------------------------------------------------
# BC (All Data)                       82.84           4.17            96.2            21.8           
# BC-Safe (Safe Data Only)            80.90           0.64            93.6            6.4            
# CBF-QP (Goal Reaching)              78.11           0.00            89.0            0.0            
# CBF-QP (BC + CBF)                   82.29           0.00            95.2            0.0            
# CBF-QP (BC-Safe + CBF)              81.04           0.00            93.4            0.0            
# CCBF-QP (Goal Reaching)             79.17           0.00            90.2            0.0            
# CCBF-QP (BC + CBF)                  82.32           0.00            95.4            0.0            
# CCBF-QP (BC-Safe + CBF)             81.14           0.00            93.0            0.0    

CBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"#cbf
CCBF_CHECKPOINT="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"##idbf



BC_MODEL_PATH = "bc_model.pt"
BC_SAFE_MODEL_PATH = "bc_safe_model.pt"
DEVICE = "cpu"

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BCTrainer:
    """Behavior Cloning Trainer for the DubinsCar environment."""
    
    def __init__(self, actor, optimizer, device="cpu"):
        """Initialize the BC trainer with an actor network and optimizer."""
        self.actor = actor
        self.optimizer = optimizer
        self.device = device
        self.loss_history = []
        
    def train_step(self, states, actions):
        """Perform a single training step using behavior cloning."""
        self.optimizer.zero_grad()
        
        # Move data to device
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)

        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        
        # Forward pass
        _, _, log_probs = self.actor(states_tensor, actions_tensor)
        
        # BC loss is negative log likelihood of expert actions
        loss = -log_probs.mean()
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs=10, log_interval=10):
        """Train the actor using behavior cloning for multiple epochs."""
        print(f"Training behavior cloning model for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for states, actions, _ in dataloader:
                loss = self.train_step(states, actions)
                epoch_loss += loss
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            self.loss_history.append(avg_epoch_loss)
            
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return self.loss_history

import numpy as np
import torch

def create_dataloader(data, batch_size=64, only_safe=False):
    """Create a dataloader from the provided dataset."""
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    costs = data["costs"]
    rewards = data["rewards"]
    episode_starts = data["episode_starts"]
    episode_lengths = data["episode_lengths"]
    
    if only_safe:
        # Identify safe trajectories (episodes with zero cost throughout)
        safe_episodes = []
        start_idx = 0
        safe_rewards = []
        safe_costs = []
        for length in episode_lengths:
            end_idx = start_idx + length
            if np.all(costs[start_idx:end_idx] == 0):
                safe_episodes.append((start_idx, end_idx))
                safe_rewards.append(np.mean(rewards[start_idx:end_idx]))
                safe_costs.append(np.mean(costs[start_idx:end_idx]))
            start_idx = end_idx
        
        # Extract states, actions, and next_states from safe trajectories
        safe_indices = np.concatenate([np.arange(start, end) for start, end in safe_episodes])
        states = states[safe_indices]
        actions = actions[safe_indices]
        next_states = next_states[safe_indices]
        
        avg_safe_reward = np.mean(safe_rewards) if safe_rewards else 0
        avg_safe_cost = np.mean(safe_costs) if safe_costs else 0
        
        print(f"Using {len(states)} safe transitions from {len(safe_episodes)} safe trajectories out of {len(episode_lengths)} total trajectories")
        print(f"Average state reward of safe trajectories: {avg_safe_reward:.4f}")
        print(f"Average state cost of safe trajectories: {avg_safe_cost:.4f}")
    else:
        # Calculate average reward and cost for all trajectories
        all_rewards = []
        all_costs = []
        start_idx = 0
        for length in episode_lengths:
            end_idx = start_idx + length
            all_rewards.append(np.mean(rewards[start_idx:end_idx]))
            all_costs.append(np.mean(costs[start_idx:end_idx]))
            start_idx = end_idx
        
        avg_all_reward = np.mean(all_rewards)
        avg_all_cost = np.mean(all_costs)
        
        print(f"Using all {len(states)} transitions from {len(episode_lengths)} trajectories")
        print(f"Average reward of all trajectories: {avg_all_reward:.4f}")
        print(f"Average cost of all trajectories: {avg_all_cost:.4f}")
    
    # Create dataset class for batching
    class SimpleDataset:
        def __init__(self, states, actions, next_states):
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.length = len(states)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return self.states[idx], self.actions[idx], self.next_states[idx]
    
    dataset = SimpleDataset(states, actions, next_states)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader


def load_dataset():
    """Load the dataset from the specified path."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        return data
    else:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

def evaluate_model(actor, env, num_episodes=5, render=False, render_delay=0.1):
    """Evaluate a trained model in the environment."""
    total_rewards = []
    total_costs = []
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        while not done and step < 200:  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(DEVICE)
                _, action, _ = actor(state_tensor, deterministic=True)
                action = action.cpu().numpy().flatten()
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            step += 1
            
            # Check for collision
            if cost > 0:
                episode_collision = True
            
            # Render if requested
            if render:
                env.render()
                time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # Close rendering window
        # if render:
        #     plt.close()
        
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        success_count += int(info["goal_reached"])
        collision_count += int(episode_collision)
        
        # Print metrics after every episode
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    # Calculate averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    success_rate = success_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100
    
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cost: {avg_cost:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    
    return avg_reward, avg_cost, success_rate, collision_rate

class CBFQPController_BC:
    """Controller that combines BC policy with CBF-QP safety filter."""
    
    def __init__(self, actor, cbf, device="cpu",alpha=0.1):
        """Initialize the controller with a policy network and CBF."""
        self.actor = actor
        self.cbf = cbf
        self.device = device
        self.dt = 0.1  # Environment timestep
        self.alpha=alpha
        
    def get_safe_action(self, state, car_dynamics_model):
        """
        Get a safe action by solving a QP that minimizes deviation from the BC policy 
        while satisfying CBF constraints using CVXPY.
        """
        # Get nominal action from behavior cloning policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, action, _ = self.actor(state_tensor, deterministic=True)
            nominal_action = action.cpu().numpy().flatten()
        
         # Current control barrier function value
        current_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        h_x = self.cbf(current_state_tensor).item()##h_x is scalar
        # Get gradient of CBF with respect to state
        current_state_tensor.requires_grad_(True)
        h_x_tensor = self.cbf(current_state_tensor)
        gradient=torch.autograd.grad(h_x_tensor,current_state_tensor,retain_graph=True)[0]#torch.Size([1, 4])
        gradient_numpy=gradient.cpu().numpy()#shape  is 1,4 
        # xdot = f(x) + g(x)u where:
        # f(x) = [0, 0, 0, 0]
        #g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])
        g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])   ##shape is 4,2

        # Decision variables (control inputs)
        u = cp.Variable(2) #shape 2,
        
        # Objective: minimize ||u - u_nominal||^2
        objective = cp.Minimize(cp.sum_squares(u - nominal_action))

        # CBF constraint: grad(h)^T g(x)u >= -alpha * h(x)
        # CBF parameter alpha ##if bigger more conservative, reacting strongly to small violations and maintaining a greater safety margin/can make the control action aggressive, potentially affecting performance or feasibility
        B_x = -self.alpha * h_x  # Right hand side of the constraint
        
        # Control limits
        control_limit = 3.0
        # Constraints
        constraints = [
            gradient_numpy @ g @ u >= B_x,  # 1,4 @  4,2 @ 2, = 1,
           # u <= control_limit,               # Upper control limits
            #u >= -control_limit               # Lower control limits
        ]
        
        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Solve using OSQP solver
            prob.solve(solver=cp.OSQP)
            
            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                safe_action = u.value
            else:
                print(f"Warning: QP could not find optimal solution (status: {prob.status}), using nominal action")
                safe_action = nominal_action
        except Exception as e:
            print(f"Error in QP solver: {e}")
            safe_action = nominal_action
        
        return safe_action, nominal_action

def evaluate_cbf_qp_controller(controller, env, num_episodes=1, render=True, render_delay=0.001):
    """Evaluate the CBF-QP controller in the environment with side-by-side visualization."""
    total_rewards = []
    total_costs = []
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        # For tracking actions and CBF values
        nominal_actions = []
        safe_actions = []
        cbf_values = []
        car_positions = []  # Track car trajectory
        
        # Set up live plotting
        if render:
            # Create a new figure for our custom plots
            # This is separate from the environment's rendering
            plt.ion()  # Turn on interactive mode
            fig = plt.figure(figsize=(10, 8))
            fig.canvas.manager.set_window_title('CBF Analysis')
            
            # Create subplots for CBF and actions
            ax2 = fig.add_subplot(3, 1, 1)
            ax2.set_title('CBF Values Over Time')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('CBF Value')
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line for CBF
            
            # Subplot for x-velocity actions
            ax3 = fig.add_subplot(3, 1, 2)
            ax3.set_title('X-Velocity Actions')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Action Value')
            
            # Subplot for y-velocity actions
            ax4 = fig.add_subplot(3, 1, 3)
            ax4.set_title('Y-Velocity Actions')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Action Value')
            
            plt.tight_layout()

            cbf_plot, = ax2.plot([], [], 'g-', label='CBF Value')
            
            nominal_x_plot, = ax3.plot([], [], 'b-', label='Nominal')
            safe_x_plot, = ax3.plot([], [], 'r-', label='Safe')
            
            nominal_y_plot, = ax4.plot([], [], 'b-', label='Nominal')
            safe_y_plot, = ax4.plot([], [], 'r-', label='Safe')
            
            # Add legends
            # ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
        
        while not done :  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get CBF value for current state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(controller.device)
                cbf_value = controller.cbf(state_tensor).item()
                cbf_values.append(cbf_value)
            
            # Get safe action from CBF-QP controller
            safe_action, nominal_action = controller.get_safe_action(full_state, env)
            
            # Store actions and positions for visualization
            nominal_actions.append(nominal_action)
            safe_actions.append(safe_action)
            car_positions.append(state[:2])  # Store car position [x, y]
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(safe_action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            if cost > 0:
                episode_collision = True
            step += 1
            
            # First render the environment (this will use the environment's built-in rendering)
            if render:
                env.render()
            
            # Update our custom plots
            if render and step % 1 == 0:  # Update every step
                # Update CBF plot
                steps_array = np.arange(len(cbf_values))
                cbf_plot.set_data(steps_array, cbf_values)
                ax2.set_xlim(0, max(20, step))
                ax2.set_ylim(min(min(cbf_values), -0.1) - 0.1, max(max(cbf_values), 0.1) + 0.1)
                # Update action plots if we have actions
                if len(nominal_actions) > 0:
                    nominal_actions_array = np.array(nominal_actions)
                    safe_actions_array = np.array(safe_actions)
                    steps_array = np.arange(len(nominal_actions))
                    # X-velocity actions
                    nominal_x_plot.set_data(steps_array, nominal_actions_array[:, 0])
                    safe_x_plot.set_data(steps_array, safe_actions_array[:, 0])
                    ax3.set_xlim(0, max(20, step))
                    ax3.set_ylim(
                        min(nominal_actions_array[:, 0].min(), safe_actions_array[:, 0].min()) - 0.5,
                        max(nominal_actions_array[:, 0].max(), safe_actions_array[:, 0].max()) + 0.5
                    )
                    # Y-velocity actions
                    nominal_y_plot.set_data(steps_array, nominal_actions_array[:, 1])
                    safe_y_plot.set_data(steps_array, safe_actions_array[:, 1])
                    ax4.set_xlim(0, max(20, step))
                    ax4.set_ylim(
                        min(nominal_actions_array[:, 1].min(), safe_actions_array[:, 1].min()) - 0.5,
                        max(nominal_actions_array[:, 1].max(), safe_actions_array[:, 1].max()) + 0.5
                    )
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # End of episode, turn off interactive mode
        if render:
            plt.ioff()

            # plt.close(traj_fig)
            plt.close(fig)
        
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        success_count += int(info["goal_reached"])
        collision_count += int(episode_collision)
        
        # print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
        #       f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    # Calculate averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    success_rate = success_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100
    
    print(f"\nCBF-QP Controller Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cost: {avg_cost:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    
    return avg_reward, avg_cost, success_rate, collision_rate

def train_all_models(data):
    """Train BC models on all data and only safe data."""
    # Define model parameters
    state_dim = 4  # car (2) + obstacle (2)
    action_dim = 2
    hidden_sizes = (64, 64)
    action_low = np.array([-3.0, -3.0])  # Based on environment limits
    action_high = np.array([3.0, 3.0])
    
    # Create dataloaders
    all_data_loader = create_dataloader(data, batch_size=64, only_safe=False)
    safe_data_loader = create_dataloader(data, batch_size=64, only_safe=True)
    
    # Create models
    print("\n" + "="*50)
    print("Training BC model on all data...")
    print("="*50)
    bc_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_optimizer = optim.Adam(bc_model.parameters(), lr=1e-4)
    bc_trainer = BCTrainer(bc_model, bc_optimizer, device=DEVICE)
    bc_loss_history = bc_trainer.train(all_data_loader, epochs=50, log_interval=5)
    
    # Save BC model
    torch.save(bc_model.state_dict(), BC_MODEL_PATH)
    print(f"BC model saved to {BC_MODEL_PATH}")
    
    print("\n" + "="*50)
    print("Training BC-Safe model on safe data only...")
    print("="*50)
    bc_safe_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_safe_optimizer = optim.Adam(bc_safe_model.parameters(), lr=1e-4)
    bc_safe_trainer = BCTrainer(bc_safe_model, bc_safe_optimizer, device=DEVICE)
    bc_safe_loss_history = bc_safe_trainer.train(safe_data_loader, epochs=50, log_interval=5)
    
    # Save BC-Safe model
    torch.save(bc_safe_model.state_dict(), BC_SAFE_MODEL_PATH)
    print(f"BC-Safe model saved to {BC_SAFE_MODEL_PATH}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(bc_loss_history, label="BC (All Data)")
    plt.plot(bc_safe_loss_history, label="BC-Safe (Safe Data Only)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("bc_training_loss.png")
    plt.close()
    
    return bc_model, bc_safe_model

def load_or_train_models(data):
    """Load models if they exist, otherwise train them."""
    # Define model parameters
    state_dim = 4  # car (2) + obstacle (2)
    action_dim = 2
    hidden_sizes = (64, 64)
    action_low = np.array([-3.0, -3.0])
    action_high = np.array([3.0, 3.0])
    
    # Check if models exist
    bc_model_exists = os.path.exists(BC_MODEL_PATH)
    bc_safe_model_exists = os.path.exists(BC_SAFE_MODEL_PATH)
    
    # Create models
    bc_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    bc_safe_model = MLPGaussianActor(
        obs_dim=state_dim,
        act_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_sizes=hidden_sizes,
        activation=nn.ReLU,
        device=DEVICE
    ).to(DEVICE)
    
    # Load models if they exist, otherwise train them
    if bc_model_exists and bc_safe_model_exists:
        # safe_data_loader = create_dataloader(data, batch_size=64, only_safe=True)
    
        print(f"Loading BC models from {BC_MODEL_PATH} and {BC_SAFE_MODEL_PATH}")
        bc_model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=DEVICE))
        bc_safe_model.load_state_dict(torch.load(BC_SAFE_MODEL_PATH, map_location=DEVICE))
    else:
        print("Training new BC models...")
        bc_model, bc_safe_model = train_all_models(data)
        
    
    return bc_model, bc_safe_model

def load_cbf_model(path):
    """Load the CBF model from checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CBF checkpoint not found at {path}")
    
    print(f"Loading CBF model from {path}")
    
    # Initialize CBF model
    cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=128).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=DEVICE)
    cbf.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    cbf.eval()
    
    return cbf

class CBFQPController_goal_reaching:
    """Controller that combines BC policy with CBF-QP safety filter."""
    
    def __init__(self, actor, cbf, device="cpu",env=None,alpha=0.1):
        """Initialize the controller with a policy network and CBF."""
        self.actor = actor
        self.cbf = cbf
        self.device = device
        self.dt = 0.1  # Environment timestep
        self.env=env
        self.alpha=alpha
    def get_safe_action(self, state, car_dynamics_model):
        """
        Get a safe action by solving a QP that minimizes deviation from the BC policy 
        while satisfying CBF constraints using CVXPY.
        """
        # Get nominal action from behavior cloning policy
        nominal_action=self.env.goal_reaching_controller()##shape 2,
        # Current control barrier function value
        current_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        h_x = self.cbf(current_state_tensor).item()##h_x is scalar
        # Get gradient of CBF with respect to state
        current_state_tensor.requires_grad_(True)
        h_x_tensor = self.cbf(current_state_tensor)
        gradient=torch.autograd.grad(h_x_tensor,current_state_tensor)[0]#torch.Size([1, 4])
        gradient_numpy=gradient.cpu().numpy()#shape  is 1,4 
        # xdot = f(x) + g(x)u where:
        # f(x) = [0, 0, 0, 0]
        #g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])
        g = np.array([[1, 0, ], [0, 1, ], [0, 0], [0, 0]])   ##shape is 4,2

        # Decision variables (control inputs)
        u = cp.Variable(2) #shape 2,
        
        # Objective: minimize ||u - u_nominal||^2
        objective = cp.Minimize(cp.sum_squares(u - nominal_action))

        # CBF constraint: grad(h)^T g(x)u >= -alpha * h(x)
          # CBF parameter ##if bigger more conservative, reacting strongly to small violations and maintaining a greater safety margin/can make the control action aggressive, potentially affecting performance or feasibility
        B_x = -self.alpha * h_x  # Right hand side of the constraint
        
        
        # Control limits
        control_limit = 3.0
        # Constraints
        constraints = [
            gradient_numpy @ g @ u >= B_x,  # 1,4 @  4,2 @ 2, = 1,
            #u <= control_limit,               # Upper control limits
            #u >= -control_limit               # Lower control limits
        ]
        
        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Solve using OSQP solver
            prob.solve(solver=cp.OSQP)
            
            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                safe_action = u.value
            else:
                print(f"Warning: QP could not find optimal solution (status: {prob.status}), using nominal action")
                safe_action = nominal_action
        except Exception as e:
            print(f"Error in QP solver: {e}")
            safe_action = nominal_action
        
        return safe_action, nominal_action

def evaluate_cbf_qp_controller_goal_reaching(controller, env, num_episodes=1, render=True, render_delay=0.001):
    """Evaluate the CBF-QP controller in the environment with side-by-side visualization."""
    total_rewards = []
    total_costs = []
    success_count = 0
    collision_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        episode_collision = False
        
        # For tracking actions and CBF values
        nominal_actions = []
        safe_actions = []
        cbf_values = []
        car_positions = []  # Track car trajectory
        
        # Set up live plotting
        if render:
            # Create a new figure for our custom plots
            # This is separate from the environment's rendering
            plt.ion()  # Turn on interactive mode
            fig = plt.figure(figsize=(10, 8))
            fig.canvas.manager.set_window_title('CBF Analysis')

            # Create subplots for CBF and actions
            ax2 = fig.add_subplot(3, 1, 1)
            ax2.set_title('CBF Values Over Time')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('CBF Value')
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Zero line for CBF
            
            # Subplot for x-velocity actions
            ax3 = fig.add_subplot(3, 1, 2)
            ax3.set_title('X-Velocity Actions')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Action Value')
            
            # Subplot for y-velocity actions
            ax4 = fig.add_subplot(3, 1, 3)
            ax4.set_title('Y-Velocity Actions')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Action Value')
            
            plt.tight_layout()
            
            # Position figures side by side on screen
            # Get screen dimensions and position windows accordingly
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                try:
                    # Try to position the windows - this will only work on some systems
                    # with certain backends like TkAgg or Qt
                    traj_fig.canvas.manager.window.wm_geometry("+0+0")
                    fig.canvas.manager.window.wm_geometry("+600+0")
                except:
                    pass  # If positioning fails, just continue

            cbf_plot, = ax2.plot([], [], 'g-', label='CBF Value')
            
            nominal_x_plot, = ax3.plot([], [], 'b-', label='Nominal')
            safe_x_plot, = ax3.plot([], [], 'r-', label='Safe')
            
            nominal_y_plot, = ax4.plot([], [], 'b-', label='Nominal')
            safe_y_plot, = ax4.plot([], [], 'r-', label='Safe')
            
            # Add legends
            # ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()

        
        while not done and step < 200:  # Cap at 200 steps to prevent infinite episodes
            # Prepare state input (append obstacle position)
            full_state = np.concatenate([state, env.obstacle_position])
            
            # Get CBF value for current state
            with torch.no_grad():
                state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(controller.device)
                cbf_value = controller.cbf(state_tensor).item()
                cbf_values.append(cbf_value)
            
            # Get safe action from CBF-QP controller
            safe_action, nominal_action = controller.get_safe_action(full_state, env)
            
            # Store actions and positions for visualization
            nominal_actions.append(nominal_action)
            safe_actions.append(safe_action)
            car_positions.append(state[:2])  # Store car position [x, y]
            
            # Take step in environment
            next_state, reward, cost, done, info = env.step(safe_action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += cost
            if cost > 0:
                episode_collision = True
            step += 1
            
            # First render the environment (this will use the environment's built-in rendering)
            if render:
                env.render()
            
            # Update our custom plots
            if render and step % 1 == 0:  # Update every step
                car_positions_array = np.array(car_positions)

                # Update CBF plot
                steps_array = np.arange(len(cbf_values))
                cbf_plot.set_data(steps_array, cbf_values)
                ax2.set_xlim(0, max(20, step))
                ax2.set_ylim(min(min(cbf_values), -0.1) - 0.1, max(max(cbf_values), 0.1) + 0.1)
                
                # Update action plots if we have actions
                if len(nominal_actions) > 0:
                    nominal_actions_array = np.array(nominal_actions)
                    safe_actions_array = np.array(safe_actions)
                    steps_array = np.arange(len(nominal_actions))
                    
                    # X-velocity actions
                    nominal_x_plot.set_data(steps_array, nominal_actions_array[:, 0])
                    safe_x_plot.set_data(steps_array, safe_actions_array[:, 0])
                    ax3.set_xlim(0, max(20, step))
                    ax3.set_ylim(
                        min(nominal_actions_array[:, 0].min(), safe_actions_array[:, 0].min()) - 0.5,
                        max(nominal_actions_array[:, 0].max(), safe_actions_array[:, 0].max()) + 0.5
                    )
                    
                    # Y-velocity actions
                    nominal_y_plot.set_data(steps_array, nominal_actions_array[:, 1])
                    safe_y_plot.set_data(steps_array, safe_actions_array[:, 1])
                    ax4.set_xlim(0, max(20, step))
                    ax4.set_ylim(
                        min(nominal_actions_array[:, 1].min(), safe_actions_array[:, 1].min()) - 0.5,
                        max(nominal_actions_array[:, 1].max(), safe_actions_array[:, 1].max()) + 0.5
                    )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
                time.sleep(render_delay)
            
            # Update state
            state = next_state
        
        # End of episode, turn off interactive mode
        if render:
            plt.ioff()
            
            # Save the final figures
            # traj_fig.savefig(f"trajectory_episode_{episode+1}.png")
            # fig.savefig(f"cbf_analysis_episode_{episode+1}.png")
            
            # plt.close(traj_fig)
            plt.close(fig)
        
        # Update metrics
        total_rewards.append(episode_reward)
        total_costs.append(episode_cost)
        success_count += int(info["goal_reached"])
        collision_count += int(episode_collision)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {episode_cost:.2f}, " +
              f"Steps = {step}, Goal reached = {info['goal_reached']}, Collision = {episode_collision}")
    
    # Calculate averages
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    success_rate = success_count / num_episodes * 100
    collision_rate = collision_count / num_episodes * 100
    
    print(f"\nCBF-QP Controller Evaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Cost: {avg_cost:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Collision Rate: {collision_rate:.1f}%")
    
    return avg_reward, avg_cost, success_rate, collision_rate

def main():
    """Main function to train, evaluate, and visualize all models."""
    # Set random seed for reproducibility
    seed_everything(42)
    
    print(f"Using device: {DEVICE}")
    
    # Load dataset
    data = load_dataset()

    # Load or train Behavior Cloning (BC) models
    bc_model, bc_safe_model = load_or_train_models(data)
    
    # Create environment for testing
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=GOAL_POSITION, obstacle_radius=4.0)

    # Initialize metrics for each model
    bc_metrics = None
    bc_safe_metrics = None
    cbf_qp_metrics = None
    
    
    num_episodes=5
    # Evaluate BC model (on all data)
    print("\n" + "="*50)
    print("Evaluating BC model (all data)...")
    print("="*50)
    bc_metrics = evaluate_model(bc_model, env, num_episodes=num_episodes, render=False, render_delay=0.000001)
    
    # Evaluate BC-Safe model (on safe data only)
    print("\n" + "="*50)
    print("Evaluating BC-Safe model (safe data only)...")
    print("="*50)
    bc_safe_metrics = evaluate_model(bc_safe_model, env, num_episodes=num_episodes, render=False, render_delay=0.000001)


    # Load the CBF model
    cbf_model = load_cbf_model(CBF_CHECKPOINT)

    # CBF-QP controller for goal-reaching with behavior cloning model
    cbf_qp_controller_goal_reaching = CBFQPController_goal_reaching(bc_model, cbf_model, device=DEVICE, env=env,alpha=1)
    print("\n" + "="*50)
    print("Evaluating nominal CBF-QP controller (goal-reaching)...")
    print("="*50)
    cbf_qp_metrics_goal_reaching = evaluate_cbf_qp_controller(cbf_qp_controller_goal_reaching, env, num_episodes=num_episodes, render=False, render_delay=0.00001)

    # CBF-QP controller using BC model and CBF
    cbf_qp_controller_bc = CBFQPController_BC(bc_model, cbf_model, device=DEVICE,alpha=1)
    print("\n" + "="*50)
    print("Evaluating CBF-QP controller (BC + CBF)...")
    print("="*50)
    cbf_qp_metrics_bc = evaluate_cbf_qp_controller(cbf_qp_controller_bc, env, num_episodes=num_episodes, render=False, render_delay=0.00001)

    # NEW: CBF-QP controller using BC-Safe model and CBF
    cbf_qp_controller_bc_safe = CBFQPController_BC(bc_safe_model, cbf_model, device=DEVICE,alpha=1)
    print("\n" + "="*50)
    print("Evaluating CBF-QP controller (BC-Safe + CBF)...")
    print("="*50)
    cbf_qp_metrics_bc_safe = evaluate_cbf_qp_controller(cbf_qp_controller_bc_safe, env, num_episodes=num_episodes, render=False, render_delay=0.00001)


    ccbf_model = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=128).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(CCBF_CHECKPOINT, map_location=DEVICE)
    ccbf_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    ccbf_model.eval()

    
    # CBF-QP controller for goal-reaching with behavior cloning model
    ccbf_qp_controller_goal_reaching = CBFQPController_goal_reaching(bc_model, ccbf_model, device=DEVICE, env=env,alpha=1)
    print("\n" + "="*50)
    print("Evaluating nominal CCBF-QP controller (goal-reaching)...")
    print("="*50)
    ccbf_qp_metrics_goal_reaching = evaluate_cbf_qp_controller(ccbf_qp_controller_goal_reaching, env, num_episodes=num_episodes, render=False, render_delay=0.00001)

    # CBF-QP controller using BC model and CBF
    ccbf_qp_controller_bc = CBFQPController_BC(bc_model, ccbf_model, device=DEVICE,alpha=1)
    print("\n" + "="*50)
    print("Evaluating CCBF-QP controller (BC + CCBF)...")
    print("="*50)
    ccbf_qp_metrics_bc = evaluate_cbf_qp_controller(ccbf_qp_controller_bc, env, num_episodes=num_episodes, render=False, render_delay=0.00001)

    # NEW: CBF-QP controller using BC-Safe model and CBF
    ccbf_qp_controller_bc_safe = CBFQPController_BC(bc_safe_model, ccbf_model, device=DEVICE,alpha=1)
    print("\n" + "="*50)
    print("Evaluating CCBF-QP controller (BC-Safe + CCBF)...")
    print("="*50)
    ccbf_qp_metrics_bc_safe = evaluate_cbf_qp_controller(ccbf_qp_controller_bc_safe, env, num_episodes=num_episodes, render=False, render_delay=0.00001)




    # Comparison of all methods
    print("\n" + "="*50)
    print(f"COMPARISON OF ALL METHODS {num_episodes} runs")
    print("="*50)
    
    # Prepare the methods and metrics for comparison
# Prepare the methods and metrics for comparison
    methods = [
        "BC (All Data)", 
        "BC-Safe (Safe Data Only)", 
        
        "CBF-QP (Goal Reaching)",
        "CBF-QP (BC + CBF)", 
        "CBF-QP (BC-Safe + CBF)", 

        "CCBF-QP (Goal Reaching)", 
        "CCBF-QP (BC + CBF)", 
        "CCBF-QP (BC-Safe + CBF)"
    ]

    all_metrics = [
        bc_metrics, 
        bc_safe_metrics, 
        cbf_qp_metrics_goal_reaching, 
        cbf_qp_metrics_bc, 
        cbf_qp_metrics_bc_safe, 

        ccbf_qp_metrics_goal_reaching, 
        ccbf_qp_metrics_bc, 
        ccbf_qp_metrics_bc_safe
    ]

    # Print the headers for the comparison table
    print(f"{'Method':<35} {'Avg Reward':<15} {'Avg Cost':<15} {'Success %':<15} {'Collision %':<15}")
    print("-" * 95)
    
    # Print the metrics for each method
    for method, metrics in zip(methods, all_metrics):
        if metrics is not None:
            avg_reward, avg_cost, success_rate, collision_rate = metrics
            print(f"{method:<35} {avg_reward:<15.2f} {avg_cost:<15.2f} {success_rate:<15.1f} {collision_rate:<15.1f}")
        else:
            print(f"{method:<35} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

if __name__ == "__main__":
    main()

