import matplotlib.pyplot as plt

def average_array(array, chunk_size=None):
    means = []
    if chunk_size is None:
        chunk_size = max((len(array) // 100, 1))

    for i in range(0, len(array), chunk_size):
        sublist = array[i:i + chunk_size]
        means.append(sum(sublist)/len(sublist))

    return means

def plot_graphs(lossX, avg_action_valueX, rewards):
    # Create a figure with two subplots next to each other
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))  # 1 row, 2 columns
    
    # Plot the first graph: Mean Loss
    chunk_size = max((len(lossX) // 100, 1))
    mean_lossX = average_array(lossX)
    steps =[i * chunk_size for i in range(len(mean_lossX))]

    axs[0].plot(steps, mean_lossX, label='Mean Loss')
    axs[0].set_title(f'Mean loss')
    axs[0].set_xlabel('Training steps')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)

    # Plot the second graph: Mean Average Action Value
    chunk_size = max((len(lossX) // 100, 1))
    mean_avg_action_valueX = average_array(avg_action_valueX)
    steps =[i * chunk_size for i in range(len(mean_lossX))]

    axs[1].plot(steps, mean_avg_action_valueX, label='Mean Avg Action Value', color='orange')
    axs[1].set_title(f'Mean action value')
    axs[1].set_xlabel('Training steps')
    axs[1].set_ylabel('Action value')
    axs[1].grid(True)

    chunk_size = 10
    mean_rewards = average_array(rewards, chunk_size)
    steps = [i * chunk_size for i in range(len(mean_rewards))]

    axs[2].plot(steps, mean_rewards, label='Mean Rewards', color='blue')
    axs[2].set_title(f'Mean reward per episode')
    axs[2].set_xlabel('Episodes')
    axs[2].set_ylabel('Reward')
    axs[2].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()