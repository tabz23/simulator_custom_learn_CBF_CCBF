import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import wandb  # Import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))  
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from modules.dataset import *
from modules.network import *
from modules.trainer_ccbf import *
from envs.car import *
from matplotlib.collections import LineCollection
import random
DATASET_PATH = "safe_rl_dataset.npz"  # Updated path to store/load dataset
goal_position=np.array([20, 21])


rng = random.Random()  # This uses a new random state
random_value = rng.randint(100, 999)
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
    wandb.log({"dataset_generation_started": True, "num_trajectories": num_trajectories})

    state_list, action_list, next_state_list = [], [], []
    reward_list, cost_list, done_list = [], [], []
    
    # Track episode boundaries for visualization
    episode_starts = [0]
    episode_lengths = []

    # Initialize with obstacle radius 0
    
    r = 0####IMP IMP IMP IMP NEXT TIME LET THIS BE 0.01 SINCE R=0 IN HE ENVIRMNT SETS R TO 4 INSTEAD
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
            wandb.log({"obstacle_radius": r, "trajectory_count": trajectory_count})

        # Calculate naive controller action
        velocity_command = env.goal_reaching_controller()
        
        # Apply CBF to make the action safe
        velocity_command_safe = env.goal_reaching__safe_controller(velocity_command)
        
        # Take a step with the safe action
        state_next, reward, cost, done, info = env.step(velocity_command_safe)
        ##SIMULATOR RETURNS STATE AS POSITION OF CAR, BUT WE APPEND OBSTACLE POSITION BEFORE ADDING IT TO THE DATASET
        
        # Store data
        state_list.append(np.concatenate([state, env.obstacle_position]))##STATE IN DATASET HAS 4 DIMENSIONS
        action_list.append(velocity_command_safe)
        next_state_list.append(np.concatenate([state_next, env.obstacle_position]))
        reward_list.append(reward)
        cost_list.append(cost)
        done_list.append(float(done))
        
        # Update state
        state = state_next
        step_count += 1
        current_episode_length += 1
        #if len ep is ep is 2(reached target in 2 steps), it contains 2 datapoints and second data point contains reward and done flag
        #trajlength is 2. if we do for length in ep_lengths then do end_idx = start_idx + length we would get end_idx is 2 and we would
        #iterate over datapt 0 and 1 where 1 has done and big reward. after we end, we set start_idx = end_idx meaning start_idx is now 2
        #since 0+eplength=2. thus now datapoint 2 corresponds to new traj
        
        # If episode is done, reset environment and track episode boundary
        if done:
            trajectory_count += 1
            episode_lengths.append(current_episode_length)
            if trajectory_count < num_trajectories:  # Only add new start if not the last trajectory
                episode_starts.append(step_count)
            current_episode_length = 0
            
            if trajectory_count % 100 == 0:
                print(f"Completed {trajectory_count} trajectories")
                wandb.log({
                    "dataset_generation_progress": trajectory_count / num_trajectories,
                    "completed_trajectories": trajectory_count,
                    "average_episode_length": np.mean(episode_lengths)
                })
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
    
    # Log dataset statistics to wandb
    wandb.log({
        "dataset_generation_completed": True,
        "dataset_size": len(state_list),
        "total_trajectories": len(episode_starts),
        "average_cost": float(np.mean(cost_list)),
        "safe_transitions_percent": float(np.mean(np.array(cost_list) == 0) * 100),
        "average_reward": float(np.mean(reward_list)),
        "average_episode_length": float(np.mean(episode_lengths))
    })

def load_dataset():
    """Loads the dataset if it exists."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        
        # Log basic dataset info to wandb
        wandb.log({
            "dataset_loaded": True,
            "dataset_size": len(data["states"]),
            "dataset_path": DATASET_PATH
        })
        
        return data
    else:
        print(f"Dataset not found at {DATASET_PATH}")
        wandb.log({"dataset_found": False, "dataset_path": DATASET_PATH})
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
        obstacle_radius = 4.0  # This appears to be constant in the dataset
        
        # Create subplot
        ax = fig.add_subplot(3, 4, plot_idx + 1)
        
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
        obstacle = plt.Circle((obstacle_position[0], obstacle_position[1]), obstacle_radius, 
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
        ax.set_aspect('equal')
        ax.set_title(f"Trajectory {traj_idx} (Length: {length})")
        
        # Add legend to first plot only
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.suptitle("Sample Trajectories from Dataset", fontsize=16, y=1.02)
    
    # Save figure and log to wandb
    plt.savefig("trajectories_visualization.png", bbox_inches='tight')
    wandb.log({"trajectories_visualization": wandb.Image("trajectories_visualization.png")})
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
    
    # Save figure and log to wandb
    plt.savefig("dataset_statistics.png", bbox_inches='tight')
    wandb.log({"dataset_statistics": wandb.Image("dataset_statistics.png")})
    
    # Log all key statistics to wandb as separate metrics
    wandb.log({
        "total_transitions": total_transitions,
        "safe_transitions": safe_transitions,
        "unsafe_transitions": unsafe_transitions,
        "safe_percentage": (safe_transitions / total_transitions) * 100,
        "total_episodes": len(episode_lengths),
        "average_episode_length": float(np.mean(episode_lengths)),
        "median_episode_length": float(np.median(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "average_reward": float(np.mean(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "unique_obstacle_positions": len(unique_obstacles)
    })
    
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
    
    wandb.log({"visualization_started": True, "num_episodes": num_episodes, "obstacle_radius": obstacle_radius})
    
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
        
        # For visualizing trajectory in wandb
        episode_positions = []
        episode_actions = []
        episode_rewards_step = []
        episode_costs_step = []
        
        while not done:
            # Calculate actions using goal-reaching controller and CBF
            naive_action = env.goal_reaching_controller()
            safe_action = env.goal_reaching__safe_controller(naive_action)
            
            # Take a step
            next_state, reward, cost, done, info = env.step(safe_action)
            episode_reward += reward
            total_cost += cost
            
            # Store step data for visualization
            episode_positions.append(env.state.tolist())
            episode_actions.append(safe_action.tolist())
            episode_rewards_step.append(reward)
            episode_costs_step.append(cost)
            
            # Display step information
            dist_to_goal = np.linalg.norm(env.goal_position - env.state)
            print(f"{step:^5} | {env.state[0]:^8.2f} | {env.state[1]:^8.2f} | {reward:^10.2f} | {cost:^6.1f} | {dist_to_goal:^10.2f}")
            
            # Render the environment
            fig = env.render()
            
            # Log environment state to wandb every few steps
            if step % 5 == 0 or done:
                plt.savefig(f"env_step_{episode}_{step}.png")
                wandb.log({
                    f"episode_{episode+1}/step_{step}": wandb.Image(f"env_step_{episode}_{step}.png"),
                    f"episode_{episode+1}/current_reward": reward,
                    f"episode_{episode+1}/cumulative_reward": episode_reward,
                    f"episode_{episode+1}/cost": cost,
                    f"episode_{episode+1}/distance_to_goal": dist_to_goal,
                })
            
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
        
        # Log episode summary to wandb
        wandb.log({
            f"episode_{episode+1}/total_steps": step,
            f"episode_{episode+1}/total_reward": episode_reward,
            f"episode_{episode+1}/total_cost": total_cost,
            f"episode_{episode+1}/goal_reached": info['goal_reached'],
            f"episode_{episode+1}/collision": info['collision'],
        })
        
        # Create and log trajectory plot to wandb
        if len(episode_positions) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            positions = np.array(episode_positions)
            ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
            ax.scatter(positions[0, 0], positions[0, 1], color='green', s=100, zorder=5, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, zorder=5, label='End')
            ax.scatter(goal_position[0], goal_position[1], color='blue', s=100, zorder=5, label='Goal')
            
            # Plot obstacle
            obstacle = plt.Circle((env.obstacle_position[0], env.obstacle_position[1]), obstacle_radius, 
                                 color='gray', alpha=0.5, label="Obstacle")
            ax.add_patch(obstacle)
            ax.set_xlim(-20, 25)
            ax.set_ylim(-20, 25)
            ax.legend()
            ax.set_aspect('equal')
            ax.set_title(f"Episode {episode+1} Trajectory")
            plt.savefig(f"episode_{episode+1}_trajectory.png")
            wandb.log({f"episode_{episode+1}/trajectory": wandb.Image(f"episode_{episode+1}_trajectory.png")})
        
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
    
    # Log overall summary to wandb
    wandb.log({
        "visualization_average_steps": float(np.mean(episode_steps)),
        "visualization_average_reward": float(np.mean(episode_rewards)),
        "visualization_average_cost": float(np.mean(episode_costs)),
        "visualization_goal_success_rate": np.mean([1 if r > 0 else 0 for r in episode_rewards]),
        "visualization_completed": True
    })
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes', 'y', '1'):
        return True
    elif v.lower() in ('false', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
def parse_args():
    parser = argparse.ArgumentParser(description="CBF Training Config")
    parser.add_argument("--num_hidden_dim", type=int, default=3, help="Number of hidden dimensions")
    parser.add_argument("--dim_hidden", type=int, default=128, help="Dimension of hidden layer")
    parser.add_argument("--use_cql_actions", type=str2bool, default=True, help="Use CQL actions or not")
    parser.add_argument("--cql_actions_weight", type=float, default=1, help="CQL actions weight")
    parser.add_argument("--num_action_samples", type=int, default=10, help="Number of action samples")
    parser.add_argument("--temp", type=float, default=1, help="Temperature parameter")
    parser.add_argument("--detach", type=str2bool, default=True, help="Detach option")
    parser.add_argument("--step_count", type=int, default=10000, help="Temperature parameter")
 
    args = parser.parse_args()
    return vars(args)  # Convert argparse Namespace to dictionary

def main():
    # Initialize wandb
    wandb.init(
        project="ccbf-car-navigation",
        name=f"{random_value}",
        config={
            "environment": "DubinsCarEnv",
            "goal_position": goal_position.tolist(),
            "max_velocity": 3.0,
            "dt": 0.1,
            "safe_distance": 5.0,
            "seed": 42,
        }
    )
    config = parse_args()
    wandb.config.update(config)
    
    seed_everything(42)
    
    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else "cpu"
    print(f"Using device: {device}")
    wandb.config.update({"device": device})

    # Load existing dataset or generate a new one
    data = load_dataset()
    if data is not None:
        print("Dataset loaded successfully")
        # Visualize dataset
        # visualize_trajectories(data, num_trajectories=12)
        # visualize_dataset_statistics(data)
    else:
        print("Generating new dataset...")
        generate_dataset(num_trajectories=2000)
        data = load_dataset()
        if data is not None:
            visualize_trajectories(data, num_trajectories=12)
            visualize_dataset_statistics(data)
    
    # Run environment visualization with real-time reward and cost display
    # run_environment_visualization(num_episodes=1, obstacle_radius=4.0, render_delay=0.1)
    
    # Your Dataset class is used here but not modified
    env = DubinsCarEnv(max_velocity=3.0, dt=0.1, goal_position=goal_position, obstacle_radius=2)
    dataset = Dataset(state_car_dim=2, state_obstacles_dim=2, control_dim=2, buffer_size=1000000, safe_distance=5.0)
    
    # Populate dataset from loaded data
    if data is not None:
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        
        print("Populating dataset object...")
        for s, a, s_next in zip(states, actions, next_states):
            dataset.add_data(s, a, s_next)
        print(f"Dataset populated with {len(states)} transitions")
        wandb.log({"dataset_populated": True, "dataset_size": len(states)})

    # Initialize CBF

    
    cbf = CBF(state_car_dim=2, state_obstacles_dim=2, dt=0.1, num_hidden_dim=config['num_hidden_dim'], dim_hidden=config['dim_hidden'])
    cbf = cbf.to(device)
    
    
    
    # Update wandb config with training parameters
    #was like this before april 27
    wandb.config.update({
        "num_hidden_dim":config['num_hidden_dim'],
        "hidden_dim":config['dim_hidden'],
        # "dim_hidden":config['dim_hidden'],
        "use_cql_actions": config['use_cql_actions'],
        
        "safe_distance": 5.0,
        "eps_safe": 0.08,
        "eps_unsafe": 0.15,
        "safe_loss_weight": 1,
        "unsafe_loss_weight": 1.2,
        "action_loss_weight": 1,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "cql_actions_weight":config['cql_actions_weight'],
  
        # "num_action_samples": 10,
        "num_state_samples": 8,
        "state_sample_std": 0.001,
        "step_count": config['step_count'],
        "num_action_samples":config['num_action_samples'],
        "temp":config['temp'],
        "detach":config['detach']
    })

    # Setup trainer with new CQL parameters
    trainer = Trainer(
        cbf, dataset, 
        safe_distance=5, 
        eps_safe=0.08, ##was 0.1
        eps_unsafe=0.15,#was 0.1
        safe_loss_weight=1, 
        unsafe_loss_weight=1.2, 
        action_loss_weight=2,
        dt=0.1, 
        batch_size=128, 
        opt_iter=1, 
        lr=1e-4, 
        device=device,
        # CQL parameters
        use_cql_actions=config['use_cql_actions'],
        cql_actions_weight=config['cql_actions_weight'],  # Weight for L_CQL_actions loss
     
        num_action_samples=config['num_action_samples'],   # Number of random actions to sample
        num_state_samples=8,     # Number of nearby states to sample
        state_sample_std=0.1,     # Standard deviation for state sampling
        detach=config['detach']
    )

    # Create lists to track metrics (extended to track CQL losses)
    losses = []
    accuracies = []
    safe_h_values = []
    unsafe_h_values = []
    cql_action_losses = []
    cql_state_losses = []
        
    # Train CBF with periodic logging and checkpoint saving
    print("Training CCBF with Conservative CQL-inspired losses...")

    
    # Create tables for detailed training logs in wandb
    wandb.define_metric("step")
    wandb.define_metric("training/*", step_metric="step")
    
    for i in range(config['step_count']):
        # Log step to wandb
        wandb.log({"step": i})
        
        # Train one step
        acc_np, loss_np, avg_safe_h, avg_unsafe_h = trainer.train_cbf()
        
        # Unpack loss components for better logging
        safe_loss, unsafe_loss, deriv_loss, cql_actions_loss, cql_states_loss = loss_np
        
        # Store metrics for later analysis
        losses.append(loss_np)
        accuracies.append(acc_np)
        safe_h_values.append(avg_safe_h)
        unsafe_h_values.append(avg_unsafe_h)
        cql_action_losses.append(cql_actions_loss)
        cql_state_losses.append(cql_states_loss)
        
        # Log metrics to wandb
        wandb.log({
            "training/acc_safe":acc_np[0],
            "training/acc_unsafe":acc_np[1],
            "training/total_loss": np.sum(loss_np),
            "training/safe_loss": safe_loss,
            "training/unsafe_loss": unsafe_loss,
            "training/deriv_loss": deriv_loss,
            "training/cql_actions_loss": cql_actions_loss,
            "training/cql_states_loss": cql_states_loss,
            "training/accuracy": acc_np,
            "training/safe_h_value": avg_safe_h,
            "training/unsafe_h_value": avg_unsafe_h,
            "training/h_value_gap": avg_safe_h - avg_unsafe_h,
            "training/progress": i / config['step_count']
        })
        
        # Save checkpoint every 1000 steps (or modify frequency as needed)
        if (i+1) % config['step_count'] == 0:
            checkpoint = {
                'step': i,
                'model_state_dict': trainer.cbf.state_dict(),
                'loss': loss_np,
                'acc': acc_np,
                'cql_params': {
                    'cql_actions_weight': trainer.cql_actions_weight,
              
                    'num_action_samples': trainer.num_action_samples,
                    'num_state_samples': trainer.num_state_samples,
                    'state_sample_std': trainer.state_sample_std
                }
            }
            checkpoint_filename = f'ccbf_ground_truth_checkpoint_{i}_{random_value}.pt'
            torch.save(checkpoint, checkpoint_filename)
            print(f"Checkpoint saved to {checkpoint_filename}")

    # Save final metrics for analysis
    metrics = {
        'losses': np.array(losses),
        'accuracies': np.array(accuracies),
        'safe_h_values': np.array(safe_h_values),
        'unsafe_h_values': np.array(unsafe_h_values),
        'cql_action_losses': np.array(cql_action_losses),
        'cql_state_losses': np.array(cql_state_losses)
    }
    torch.save(metrics, 'ccbf_training_metrics.pt')
    
    print("CCBF Training complete")

if __name__ == "__main__": 
    main()