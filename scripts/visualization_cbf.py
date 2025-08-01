import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from modules.dataset import *
from modules.network import *
from modules.trainer_ccbf import *
from envs.car import *
from matplotlib.collections import LineCollection

DATASET_PATH = "safe_rl_dataset.npz"  # Updated path to store/load dataset

##below are only varying w_cql while keeping t and actions sampled
# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_states_weight0.001_cql_actions_weight0.001_801.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_states_weight0.001_cql_actions_weight0.01_277.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.1_757.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight1_495.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight10_885.pt"

##below are only varying w_cql while keeping t and actions sampled and with DETACHED
# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.001_408.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.01_220.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight0.1_197.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight1_439.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_checkpoint_4999_cql_actionsTrue_cql_actions_weight10_562.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_7999_104.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_7999_297.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_7999_513.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_980.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_7999_730.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_748.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_226.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_679.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_980.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_965.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_349.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_712.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_557.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_334.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_841.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_291.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_445.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_325.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_325.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_4999_325.pt"

##below are varying w_cql

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_527.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_967.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_803.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_565.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_981.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_642.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_446.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_600.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_773.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_311.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_815.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_630.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_563.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_863.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_468.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_630.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_563.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_9999_159.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_19999_182.pt"

###for temp of 1 and good cql graphs below. nice stuff


# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_272.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_230.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_682.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_775.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_719.pt"

##cqls with detach for paper used

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_802.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_749.pt"##310 kenit
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_119.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"

#cqls without detach
# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_624.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_245.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_997.pt"##310 kenit
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_584.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_923.pt"


# ##idbfs
# CBF_1="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_645.pt"#p=0.0001
# CBF_2="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_894.pt"#p=0.00001
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_318.pt"#p=0.000001
# CBF_4="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_582.pt"#p=0.0000001
# CBF_5="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"#p=0.00000001

# ##idbfs 2 for paper
# CBF_1="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_927.pt"#p=0.1  10^-1
# CBF_2="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_645.pt"#p=0.0001 10^-4
# # CBF_2="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_645.pt"#p=0.00001
# # CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_318.pt"#p=0.000001
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_894.pt"#p=0.00001 10^-5
# CBF_4="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_582.pt"#p=0.0000001 10^-7
# CBF_5="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_625.pt"#p=0.00000001 10^-8

# visualization__625__645__894__318__582
# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_927.pt"#p=0.00000001


# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_961.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_386.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_927.pt"#p=0.1 eps_unsafe=0.2  w_unsafe=1.5  / safe buffer has dimensions:  214587 unsafe buffer has dimensions : 796801

##idbf new
# CBF_1="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_159.pt"#p=0.1 epsunsafe=0.1 w_unsafe=1 1munsafe
# CBF_2="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_927.pt"#p=0.1 eps_unsafe=0.2  w_unsafe=1.5  / safe buffer has dimensions:  214587 unsafe buffer has dimensions : 796801
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_853.pt"#p=0.1 eps_unsafe=0.2  w_unsafe=2  / safe buffer has dimensions:  214587 unsafe buffer has dimensions : 796801






##debugging
# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_537.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_787.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_839.pt"
# CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_500.pt"
# CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_793.pt"

# CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_537.pt"
# CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_793.pt"
# CBF_3="/Users/i.k.tabbara/Documents/python directory/IDBF_ccbf_checkpoint_927.pt"#p=0.00000001


#debugging
CBF_1="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_945.pt"
CBF_2="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_260.pt"
CBF_3="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_313.pt"
CBF_4="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_816.pt"
CBF_5="/Users/i.k.tabbara/Documents/python directory/ccbf_ground_truth_checkpoint_24999_453.pt"

goal_position=np.array([20, 21])

def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def generate_dataset(num_trajectories=2000):
    """Generates and saves a dataset of trajectories with rewards and costs."""
    print(f"Generating dataset with {num_trajectories} trajectories...")

    state_list, action_list, next_state_list = [], [], []
    reward_list, cost_list, done_list = [], [], []
    
    # Track episode boundaries for visualization
    episode_starts = [0]
    episode_lengths = []

    # Initialize with obstacle radius 0
    r = 0
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=r)
    state = env.reset()
    
    trajectory_count = 0
    step_count = 0
    current_episode_length = 0
    
    while trajectory_count < num_trajectories:
        # Increase obstacle radius every while, this takes effect in the envirnmnt generating trajectories as now CBF sees obstacle with bigger radius, generating more diverse data
        if (step_count % 25000) == 0 and step_count > 0 and step_count>0:
            r = min(r + 0.2, 5)  # Cap radius at 6 to prevent impossible scenarios
            env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=r)
            state = env.reset()
            print(f"traj {trajectory_count}: Increased obstacle radius to {r}")

        # Calculate naive controller action
        velocity_command = env.goal_reaching_controller()
        
        # Apply CBF to make the action safe
        velocity_command_safe = env.goal_reaching__safe_controller(velocity_command)
        
        # Take a step with the safe action
        state_next, reward, cost, done, info = env.step(velocity_command_safe)
        
        # Store data
        state_list.append(np.concatenate([state, env.obstacle_position]))
        action_list.append(velocity_command_safe)
        next_state_list.append(np.concatenate([state_next, env.obstacle_position]))
        reward_list.append(reward)
        cost_list.append(cost)
        done_list.append(float(done))
        
        # Update state
        state = state_next
        step_count += 1
        current_episode_length += 1
        
        # If episode is done, reset environment and track episode boundary
        if done:
            trajectory_count += 1
            episode_lengths.append(current_episode_length)
            if trajectory_count < num_trajectories:  # Only add new start if not the last trajectory
                episode_starts.append(step_count)
            current_episode_length = 0
            
            if trajectory_count % 100 == 0:
                print(f"Completed {trajectory_count} trajectories")
            state = env.reset()
    
    # Convert to numpy arrays and save
    np.savez(
        DATASET_PATH, 
        states=np.array(state_list), 
        actions=np.array(action_list), 
        next_states=np.array(next_state_list),
        rewards=np.array(reward_list),
        costs=np.array(cost_list),
        dones=np.array(done_list),
        episode_starts=np.array(episode_starts),
        episode_lengths=np.array(episode_lengths)
    )
    print(f"Dataset saved to {DATASET_PATH}")
    print(f"Dataset shape: {len(state_list)} transitions")
    print(f"Total trajectories: {len(episode_starts)}")
    print(f"Average cost per transition: {np.mean(cost_list):.4f}")

def load_dataset():
    """Loads the dataset if it exists."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        return data
    else:
        print(f"Dataset not found at {DATASET_PATH}")
        return None

def visualize_trajectories(data, num_trajectories=10):
    """Visualize selected trajectories from the dataset with improved clarity."""
    # Determine episode boundaries
    if "episode_starts" in data and "episode_lengths" in data:
        # Use pre-computed episode boundaries if available
        episode_starts = data["episode_starts"]
        episode_lengths = data["episode_lengths"]
    else:
        # Otherwise reconstruct from done flags
        dones = data["dones"]
        episode_starts = [0]
        episode_lengths = []
        
        current_length = 0
        for i in range(len(dones)):
            current_length += 1
            if dones[i] == 1.0:
                episode_lengths.append(current_length)
                if i < len(dones) - 1:  # Don't add start if last episode
                    episode_starts.append(i + 1)
                current_length = 0
    
    # Select random trajectories to visualize
    import random
    num_episodes = len(episode_starts)
    print(f"Dataset contains {num_episodes} trajectories")
    
    selected_indices = random.sample(range(num_episodes), min(num_trajectories, num_episodes))
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))
    
    # Goal position is fixed in this environment
    
    # Add a colormap for trajectory visualization
    cmap = plt.cm.viridis
    
    for plot_idx, traj_idx in enumerate(selected_indices):
        # Get start and length of this trajectory
        start_idx = episode_starts[traj_idx]
        length = episode_lengths[traj_idx] if traj_idx < len(episode_lengths) else len(data["states"]) - start_idx
        
        # Extract trajectory data
        states = data["states"][start_idx:start_idx + length]
        
        # Get car positions and obstacle positions
        car_positions = states[:, :2]  # First 2 elements are car position
        obstacle_position = states[0, 2:4]  # Elements 2-3 are obstacle position (constant within trajectory)
        # obstacle_radius = 5.0  # This appears to be constant in the dataset
        
        # Create subplot
        ax = fig.add_subplot(1, 1, plot_idx + 1)
        
        # Plot trajectory with color gradient to show direction
        points = np.array([car_positions[:, 0], car_positions[:, 1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, length)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.8)

        lc.set_array(np.arange(length))
        line = ax.add_collection(lc)
        
        # Mark start and end points
        ax.scatter(car_positions[0, 0], car_positions[0, 1], color='green', s=50, zorder=5, label="Start")
        ax.scatter(car_positions[-1, 0], car_positions[-1, 1], color='red', s=50, zorder=5, label="End")
        
        # Plot obstacle
        obstacle = plt.Circle((obstacle_position[0], obstacle_position[1]), 4, 
                             color='gray', alpha=0.5, label="Obstacle")
        ax.add_patch(obstacle)
        
        # Plot goal
        goal_size = 1.5  # Size of goal region
        goal_rect = plt.Rectangle((goal_position[0] - goal_size, goal_position[1] - goal_size), 
                                 width=2*goal_size, height=2*goal_size, 
                                 color='blue', alpha=0.3, label="Goal")
        ax.add_patch(goal_rect)
        
        # Set plot properties
        ax.set_xlim(-20, 25)
        ax.set_ylim(-20, 25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_aspect('equal')
        # ax.set_title(f"Trajectory {traj_idx} (Length: {length})")
        
        # Add legend to first plot only
        if plot_idx == 0:
           ax.legend(loc='upper left', fontsize=30)  # Larger than 'xx-large'

    
    plt.tight_layout()
    # plt.suptitle("Sample Trajectories from Dataset", fontsize=16, y=1.02)
    plt.savefig("trajectories_visualization.png", bbox_inches='tight')
    plt.show()

def visualize_dataset_statistics(data):
    """Visualize comprehensive dataset statistics."""
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    costs = data["costs"]
    dones = data["dones"]
    
    # Calculate statistics
    total_transitions = len(states)
    safe_transitions = np.sum(costs == 0)
    unsafe_transitions = np.sum(costs > 0)
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # 1. Plot reward distribution
    plt.subplot(2, 3, 1)
    plt.hist(rewards, bins=50, color='blue', alpha=0.7)
    plt.title("Reward Distribution")
    plt.xlabel("Reward Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # 2. Plot cost distribution
    plt.subplot(2, 3, 2)
    labels = ['Safe (cost=0)', 'Unsafe (cost>0)']
    counts = [safe_transitions, unsafe_transitions]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
    plt.axis('equal')
    plt.title("Safety Distribution")
    
    # 3. Plot car positions
    plt.subplot(2, 3, 3)
    car_positions = states[:, :2]
    plt.hexbin(car_positions[:, 0], car_positions[:, 1], gridsize=50, cmap='CMRmap_r')
    plt.colorbar(label="Density")
    plt.title("Car Position Distribution")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # 4. Plot action distribution
    plt.subplot(2, 3, 4)
    plt.hist2d(actions[:, 0], actions[:, 1], bins=30, cmap='Greens')
    plt.colorbar(label="Frequency")
    plt.title("Action Distribution")
    plt.xlabel("Action Dimension 1")
    plt.ylabel("Action Dimension 2")
    
    # 5. Plot obstacle positions
    plt.subplot(2, 3, 5)
    obstacle_positions = states[:, 2:4]
    unique_obstacles = np.unique(obstacle_positions, axis=0)
    plt.scatter(unique_obstacles[:, 0], unique_obstacles[:, 1], 
               c=range(len(unique_obstacles)), cmap='rainbow', s=100, alpha=0.7)
    plt.title(f"Unique Obstacle Positions\n({len(unique_obstacles)} positions)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    
    # 6. Plot episodes information
    plt.subplot(2, 3, 6)
    episode_ends = np.where(dones == 1)[0] + 1
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts
    
    plt.hist(episode_lengths, bins=20, color='purple', alpha=0.7)
    plt.title("Episode Length Distribution")
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    # Add overall title and summary
    plt.suptitle("Dataset Statistics Overview", fontsize=16, y=0.98)
    plt.figtext(0.5, 0.01, f"Total Transitions: {total_transitions} | Episodes: {len(episode_lengths)} | Avg Episode Length: {np.mean(episode_lengths):.1f}", 
               ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("dataset_statistics.png", bbox_inches='tight')
    plt.show()

def run_environment_visualization(num_episodes=5, obstacle_radius=4.0, render_delay=0.1):
    """
    Run the environment with visualization and print rewards and costs.
    
    Args:
        num_episodes: Number of episodes to run
        obstacle_radius: Radius of the obstacle
        render_delay: Delay between steps for visualization (seconds)
    """
    print("\n" + "="*50)
    print("RUNNING ENVIRONMENT VISUALIZATION")
    print("="*50)
    
    # Create environment
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=obstacle_radius)
    
    # Stats tracking
    episode_rewards = []
    episode_costs = []
    episode_steps = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print("-" * 40)
        
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0
        total_cost = 0
        
        # Create a table header for the step data
        print(f"{'Step':^5} | {'X Pos':^8} | {'Y Pos':^8} | {'Reward':^10} | {'Cost':^6} | {'Goal Dist':^10}")
        print("-" * 60)
        
        while not done:
            # Calculate actions using goal-reaching controller and CBF
            naive_action = env.goal_reaching_controller()
            safe_action = env.goal_reaching__safe_controller(naive_action)
            
            # Take a step
            next_state, reward, cost, done, info = env.step(safe_action)
            episode_reward += reward
            total_cost += cost
            
            # Display step information
            dist_to_goal = np.linalg.norm(env.goal_position - env.state)
            print(f"{step:^5} | {env.state[0]:^8.2f} | {env.state[1]:^8.2f} | {reward:^10.2f} | {cost:^6.1f} | {dist_to_goal:^10.2f}")
            
            # Render the environment
            env.render()
            
            # Add a small delay to make visualization easier to follow
            time.sleep(render_delay)
            
            step += 1
            
            # Break if taking too many steps
            if step >= 200:
                print("Episode timeout reached")
                break
                
        # Close the plot for the episode
        plt.close()
        
        # Track statistics
        episode_rewards.append(episode_reward)
        episode_costs.append(total_cost)
        episode_steps.append(step)
        
        # Display episode summary
        print("-" * 40)
        print(f"Episode {episode+1} Summary:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Cost: {total_cost:.2f}")
        print(f"  Goal Reached: {info['goal_reached']}")
        print(f"  Collisions: {info['collision']}")
    
    # Display overall summary
    print("\n" + "="*50)
    print("VISUALIZATION SUMMARY")
    print("="*50)
    print(f"Average Episode Steps: {np.mean(episode_steps):.1f}")
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f}")
    print(f"Average Episode Cost: {np.mean(episode_costs):.2f}")
    print("="*50)
    
def visualize_trajectories_and_cbf(data, cbf_paths, resolution=10000, device="mps"):
    """
    Visualize the density of all car trajectories and multiple CBF values.
    
    Args:
    data: The dataset containing trajectories
    cbf_paths: List of paths to trained CBF models (exactly 4 paths expected)
    resolution: Resolution of the CBF value heatmap
    device: Device to run the CBF models on
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load all CBF models
    cbf_models = []
    cbf_names = []
    for path in cbf_paths:
        cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=128).to(device)
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        cbf.load_state_dict(checkpoint['model_state_dict'],device)
        cbf_models.append(cbf)
        # Extract identifying part of the filename for plot titles
        cbf_names.append(path[-7:-3])
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, 6, figsize=(20, 15))
    
    # Plot trajectory density (all trajectories)
    ax_traj = axes[0]
    
    # Extract all car positions from all trajectories
    all_car_positions = data["states"][:, :2]
    
    
    ##new
    # Compute the interquartile range (IQR) for outlier detection
    x_positions = data["states"][:, 0]
    y_positions = data["states"][:, 1]

    # Compute Q1, Q3, and IQR
    lower_x, upper_x = np.percentile(x_positions, [0, 95])
    lower_y, upper_y = np.percentile(y_positions, [0, 95])

    # Filter out outliers
    mask = (x_positions >= lower_x) & (x_positions <= upper_x) & (y_positions >= lower_y) & (y_positions <= upper_y)
    filtered_positions = data["states"][mask, :2]

    # Create a hexbin plot without outliers
    h = ax_traj.hexbin(filtered_positions[:, 0], filtered_positions[:, 1], gridsize=100, cmap='CMRmap_r', mincnt=1)
    bar=plt.colorbar(h, ax=ax_traj, shrink=0.15) 
    bar.ax.tick_params(labelsize=15)  # adjust 20 to your desired size
    ##new
    # h = ax_traj.hexbin(all_car_positions[:, 0], all_car_positions[:, 1], 
    #                  gridsize=50, cmap='CMRmap_r', mincnt=1)
    # plt.colorbar(h, ax=ax_traj, label="Trajectory Density")
    
    # Add obstacle from the first state (assuming it doesn't move)
    obstacle_position = data["states"][0, 2:4]
    obstacle = plt.Circle((obstacle_position[0], obstacle_position[1]), 5.0, color='gray', alpha=0.4,label='Obstacle')
    ax_traj.add_patch(obstacle)
    
    # Add goal position
    goal_size = 1
    goal_rect = plt.Rectangle((goal_position[0] - goal_size, goal_position[1] - goal_size), 
                             width=2*goal_size, height=2*goal_size, 
                             color='blue', alpha=0.3,label='Goal')
    ax_traj.add_patch(goal_rect)
    
    # ax_traj.set_xlim(-20, 25)
    # ax_traj.set_ylim(-20, 25)
    ax_traj.set_aspect('equal')
    ax_traj.set_xticks(np.linspace(-20, 25, 4))
    # ax_traj.set_title("Density of Trajectories")
    # ax_traj.set_xlabel("X Position")
    # ax_traj.set_ylabel("Y Position")
    # ax_traj.legend(loc="upper left",fontsize=16,frameon=False)  # UNCOMMENT FOR MAIN FIGURE

    
    ax_traj.tick_params(axis='both', which='major', labelsize=16)  

    # Plot CBF values for each model
    for i, (cbf, name) in enumerate(zip(cbf_models, cbf_names)):
        ax_cbf = axes[i + 1]
        x = np.linspace(-20, 25, resolution)
        y = np.linspace(-20, 25, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Use the first obstacle position for CBF evaluation
        obstacle_pos = data["states"][0, 2:4]
        
        cbf_values = np.zeros((resolution, resolution))
        # Compute percentiles


        for j in range(resolution):
            for k in range(resolution):
                state = np.array([X[j, k], Y[j, k], obstacle_pos[0], obstacle_pos[1]])
                with torch.no_grad():
                    cbf.eval()
                    cbf_value = cbf(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).item()
                cbf_values[j, k] = cbf_value
                
        lower_bound, upper_bound = np.percentile(cbf_values, [0, 100])###added this
        # cbf_values = np.clip(cbf_values, lower_bound, upper_bound)###added this
        
        import matplotlib.colors as mcolors
        
        ##aded below
        # Compute min and max
        vmin = np.min(cbf_values)
        vmax = np.max(cbf_values)

        # Ensure vcenter=0 is only used if vmin < 0 and vmax > 0
        if vmin < 0 and vmax > 0:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            ticks = [vmin, 0, vmax]  # Set ticks at min, 0, and max
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            ticks = [vmin, vmax]  # Only min and max if 0 is not within range
                ##aded below
        
        im = ax_cbf.imshow(cbf_values, extent=[-20, 25, -20, 25], origin='lower',cmap='coolwarm',norm=norm, aspect='equal')#coolwarm
        cbar=plt.colorbar(im, ax=ax_cbf, shrink=0.15)# shrink=0.15 for 5 plots. 0.24 for 3 plots
        # Set evenly spaced ticks
        num_ticks = 5  # Adjust this to control the number of ticks
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])
        cbar.ax.tick_params(labelsize=17)  # adjust 20 to your desired size
        # ax_cbf.set_title(f"CBF Values - {name}")
        # ax_cbf.set_xlabel("X Position")
        # ax_cbf.set_ylabel("Y Position")
        
        # Add obstacle and goal to CBF plot
        
        obstacle = plt.Circle((obstacle_pos[0], obstacle_pos[1]), 5.0, color='gray', alpha=0.4)##made radius a little
        ax_cbf.add_patch(obstacle)
        goal_rect = plt.Rectangle((goal_position[0] - goal_size, goal_position[1] - goal_size), 
                                width=2*goal_size, height=2*goal_size, 
                                color='blue', alpha=0.3)
        ax_cbf.add_patch(goal_rect)
        ax_cbf.axis("off")##ADDED THIS
    # Create a custom filename based on all CBF names
    filename = f"visualization_{'_'.join(cbf_names)}.png"
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def main():
    seed_everything(42)
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print(f"Using device: {device}")

    # Load existing dataset or generate a new one
    data = load_dataset()
    if data is not None:
        print("Dataset loaded successfully")
        # Visualize dataset
        # visualize_trajectories(data, num_trajectories=12)
        # visualize_dataset_statistics(data)
        
        # cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=32).to(device)
        # # Load checkpoint
        # checkpoint = torch.load(CBF_TO_VISUALIZE, map_location=device)
        # cbf.load_state_dict(checkpoint['model_state_dict'],device)
        
        # ccbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=3, dim_hidden=32).to(device)
        # # Load checkpoint
        # checkpoint = torch.load(CCBF_TO_VISUALIZE, map_location=device)
        # ccbf.load_state_dict(checkpoint['model_state_dict'],device)
        
        
        # ccbf.eval()
        # cbf.eval()
        
        # visualize_trajectories(data, num_trajectories=1)
        visualize_trajectories_and_cbf(data, [CBF_1,CBF_2,CBF_3,CBF_4,CBF_5], resolution=110)
        # for i in range(100,200):
        #     state = torch.tensor(data["states"][i, :], dtype=torch.float32, device=device).reshape(1, -1)
        #     print(state)
        #     custom_state = torch.tensor([10,11, 10,11], dtype=torch.float32, device=device).reshape(1, -1)

        # # Compute and print the CBF value for the given state
        #     cbf_value = ccbf(state)
        #     print(cbf_value)
        
    else:
        print("Generating new dataset...")
        generate_dataset(num_trajectories=2000)
        data = load_dataset()
        if data is not None:
            visualize_trajectories(data, num_trajectories=12)
            visualize_dataset_statistics(data)
    

if __name__ == "__main__": 
    main()
    
