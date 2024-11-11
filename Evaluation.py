import matplotlib.pyplot as plt

def average_array(array, chunk_size=None):
    means = []
    if chunk_size is None:
        chunk_size = max((len(array) // 100, 1))

    for i in range(0, len(array), chunk_size):
        sublist = array[i:i + chunk_size]
        means.append(sum(sublist)/len(sublist))

    return means

def plot_graphs(lossX, avg_action_valueX):
    mean_lossX = average_array(lossX)
    mean_avg_action_valueX = average_array(avg_action_valueX)
    chunk_size = max((len(lossX) // 100, 1))
    steps =[i * chunk_size for i in range(len(mean_lossX))]

    # Create a figure with two subplots next to each other
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
    
    # Plot the first graph: Mean Loss
    axs[0].plot(steps, mean_lossX, label='Mean Loss')
    axs[0].set_title(f'Mean loss over training steps')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    # axs[0].legend()

    # Plot the second graph: Mean Average Action Value
    axs[1].plot(steps, mean_avg_action_valueX, label='Mean Avg Action Value', color='orange')
    axs[1].set_title(f'Mean action value over training steps')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Action Value')
    axs[1].grid(True)
    # axs[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()