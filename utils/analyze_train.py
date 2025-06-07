import json
import argparse
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from pathlib import Path

# Font settings to match save_spectrograms
BIGGER_SIZE = 10
SMALLER_SIZE = 8

plt.rc('font', size=BIGGER_SIZE, family='serif')  # controls default text sizes
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# Set a better figure DPI for higher quality output
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSON history file')
    parser.add_argument('json_path', type=str, help='Path to the JSON history file')
    parser.add_argument('metric', type=str, help='Metric to track (e.g., "d_loss", "g_loss", "pesq")')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output directory path for the plot. If not specified, saves to "outputs" directory')
    parser.add_argument('--name', '-n', type=str, default=None,
                      help='Custom name for the output file (without extension). If not specified, uses the metric name')
    parser.add_argument('--title', type=str, default=None,
                      help='Custom plot title. If not specified, uses a generated title')
    parser.add_argument('--xlabel', type=str, default='Epochs',
                      help='Label for the x-axis. Default: "Epochs"')
    parser.add_argument('--ylabel', type=str, default=None,
                      help='Label for the y-axis. If not specified, uses the metric name')
    parser.add_argument('--legend', type=str, nargs='+', default=None,
                      help='Custom legend labels. Provide two space-separated values for train and validation')
    parser.add_argument('--font-size', type=int, default=12,
                      help='Base font size for the plot. Default: 12')
    return parser.parse_args()

def load_history(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def ensure_dir(directory):
    """Ensure that the specified directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def plot_metric(history, metric, output_dir="outputs", output_name=None, 
               title=None, xlabel='Epochs', ylabel=None, legend=None, font_size=12):
    train_metrics = []
    val_metrics = []
    
    # Update font sizes consistently
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('legend', fontsize=font_size)
    
    for epoch in history:
        # Get training metric if it exists
        if 'train' in epoch and metric in epoch['train']:
            train_metrics.append(epoch['train'][metric])
        
        # Get validation metric if it exists
        if 'valid' in epoch and metric in epoch['valid']:
            val_metrics.append(epoch['valid'][metric])
    
    if not train_metrics:
        raise ValueError(f"Metric '{metric}' not found in training data")
    
    epochs = range(1, len(train_metrics) + 1)
    
    # Create figure with adjusted size and DPI
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Set default legend labels
    train_label = legend[0] if legend and len(legend) > 0 else f'Train {metric}'
    val_label = legend[1] if legend and len(legend) > 1 else f'Validation {metric}'
    
    # Plot training data
    plt.plot(epochs, train_metrics, 'b-', label=train_label, linewidth=2)
    
    # Plot validation data if available
    if val_metrics:
        plt.plot(epochs[:len(val_metrics)], val_metrics, 'r-', label=val_label, linewidth=2)
    
    # Set title with Unicode subscript for A_{100}
    title_text = (title or f'Training and Validation {metric}').replace('$A_{100}$', 'A₁₀₀')
    plt.title(title_text, fontsize=font_size+2, pad=20)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel or metric, fontsize=font_size)
    
    # Customize legend
    legend = plt.legend(frameon=True, fontsize=font_size-1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    
    # Customize grid and ticks
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=font_size-1)
    plt.yticks(fontsize=font_size-1)
    
    # Adjust layout to prevent title and labels from being cut off
    plt.tight_layout()
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Create output filename
    filename = f"{output_name}.png" if output_name else f"{metric}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to: {os.path.abspath(output_path)}")

def main():
    args = parse_arguments()
    
    try:
        history = load_history(args.json_path)
        output_dir = args.output if args.output else "outputs"
        plot_metric(
            history=history,
            metric=args.metric,
            output_dir=output_dir,
            output_name=args.name,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            legend=args.legend,
            font_size=args.font_size
        )
    except FileNotFoundError:
        print(f"Error: File {args.json_path} not found")
    except json.JSONDecodeError:
        print(f"Error: {args.json_path} is not a valid JSON file")
    except KeyError as e:
        print(f"Error: Metric '{args.metric}' not found in the history file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()