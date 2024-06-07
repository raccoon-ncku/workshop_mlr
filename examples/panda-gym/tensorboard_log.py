import os
from tensorboard.backend.eventlog import EventLogReader

def plot_tensorboard(log_dir, tag_name, output_file=None):
  """
  Plots data from a specific tag in a TensorBoard log directory.

  Args:
      log_dir: Path to the TensorBoard log directory.
      tag_name: Name of the tag to plot (e.g., 'loss', 'accuracy').
      output_file: Optional path to save the plot as an image (e.g., 'plot.png').
  """

  # Check if log directory exists
  if not os.path.exists(log_dir):
    raise ValueError(f"Log directory not found: {log_dir}")

  # Create EventLogReader object
  reader = EventLogReader(log_dir)

  # Extract data for the specified tag
  steps = []
  values = []
  for event in reader.AllEvents():
    if event.summary.value[0].tag == tag_name:
      steps.append(event.step)
      values.append(event.summary.value[0].simple_value)

  # Import plotting library (choose one based on your preference)
  # You can replace 'matplotlib' with other libraries like 'pyplot' or 'seaborn'
  import matplotlib.pyplot as plt

  # Configure and generate plot
  plt.plot(steps, values)
  plt.xlabel('Step')
  plt.ylabel(tag_name)
  plt.grid(True)

  if output_file:
    plt.savefig(output_file)
  else:
    plt.show()

# Example usage
log_dir = 'examples/panda-gym/tensorboard/TQC_1'  # Replace with your actual log directory
tag_name = 'loss'  # Replace with the tag you want to plot
output_file = 'loss_plot.png'  # Optional, set filename to save the plot

plot_tensorboard(log_dir, tag_name, output_file)