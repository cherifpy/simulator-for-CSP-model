import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.set_loglevel (level = 'warning')
logging.getLogger('PIL').setLevel(logging.WARNING)

def plot_gantt_chart(task_data, num_nodes, title="Task Execution and Data Transfers"):
    """Visualize the Gantt chart."""
    _, ax = plt.subplots(figsize=(12, 8))
    colors = {'processing': 'blue', 'transfer': 'red'}

    bar_height = 0.3  # Reduced height to fit both bars in one row

    for entry in task_data:
        node_id = entry['node_id']
        start = entry['start']
        end = entry.get('end', start + 1)  # Default to 1 unit if no end time
        task_type = entry['type']
        color = colors.get(task_type, 'gray')

        # Align processing and transfer bars tightly within the same node row
        if task_type == 'processing':
            y_offset = node_id + bar_height / 2
            label = str(entry['job_id'])+"-"+str(entry['task_id']) # Task ID for processing tasks
        elif task_type == 'transfer':
            y_offset = node_id - bar_height / 2
            label = f"D-{str(entry['job_id'])}"  # Job ID for data transfers
        else:
            y_offset = node_id
            label = ""

        # Plot the bar
        ax.barh(
            y=y_offset,
            width=end - start,
            left=start,
            height=bar_height,
            color=color,
            edgecolor='black'
        )

        # Center the label inside the bar
        if label:  # Only add the label if it exists
            ax.text(
                x=start + (end - start) / 2,  # Center the label horizontally
                y=y_offset,  # Align with the bar's y-offset
                s=label,
                color='white' if task_type == 'processing' else 'black',
                ha='center',  # Center align horizontally
                va='center',  # Center align vertically
                fontsize=8
            )

    # Adjust y-axis ticks and labels to reflect node IDs
    ax.set_yticks(range(num_nodes))
    ax.set_yticklabels([f"Node-{i}" for i in range(num_nodes)])

    # Add legend
    handles = [
        mpatches.Patch(color=colors['processing'], label='Processing'),
        mpatches.Patch(color=colors['transfer'], label='Dataset Transfer'),
    ]
    ax.legend(handles=handles)

    ax.set_xlabel("Time")
    ax.set_ylabel("Compute Nodes")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_line(data, title, xlabel, ylabel):
    for line in data:
        plt.plot([x[0] for x in data[line]], [x[1] for x in data[line]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
