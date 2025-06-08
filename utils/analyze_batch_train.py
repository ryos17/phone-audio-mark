import json
import argparse
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Font settings to match save_spectrograms
BIGGER_SIZE = 10
SMALLER_SIZE = 8

plt.rc('font', size=BIGGER_SIZE, family='serif')  # controls default text sizes
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=SMALLER_SIZE)
plt.rc('legend', fontsize=BIGGER_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Set a better figure DPI for higher quality output
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot training metrics from multiple JSON history files')
    parser.add_argument('json_paths', type=str, nargs='+', 
                      help='Paths to the JSON history files')
    parser.add_argument('metric', type=str, 
                      help='Metric to track (e.g., "d_loss", "g_loss", "pesq")')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output directory path for the plot. Default: "outputs"')
    parser.add_argument('--name', '-n', type=str, default=None,
                      help='Custom name for the output file (without extension)')
    parser.add_argument('--title', type=str, default=None,
                      help='Custom plot title')
    parser.add_argument('--xlabel', type=str, default='Epochs',
                      help='Label for the x-axis. Default: "Epochs"')
    parser.add_argument('--ylabel', type=str, default=None,
                      help='Label for the y-axis')
    parser.add_argument('--legend', type=str, nargs='+', default=None,
                      help='Custom legend labels (one per input file)')
    parser.add_argument('--font-size', type=int, default=12,
                      help='Base font size for the plot. Default: 12')
    parser.add_argument('--line-styles', type=str, nargs='+', default=['-', '--', '-.'],
                      help='Line styles for different runs')
    parser.add_argument('--colors', type=str, nargs='+', 
                      default=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                      help='Colors for different runs')
    parser.add_argument('--epoch-limits', type=int, nargs='+', default=None,
                      help='Maximum epochs to plot for each model (one per input file)')
    return parser.parse_args()

def load_history(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r') as f:
        return json.load(f)

def ensure_dir(directory: str) -> None:
    """Ensure that the specified directory exists, create it if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def plot_metrics(
    histories: List[Dict[str, Any]], 
    labels: List[str],
    metric: str,
    output_dir: str = "outputs",
    output_name: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = 'Epochs',
    ylabel: Optional[str] = None,
    font_size: int = 12,
    line_styles: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    epoch_limits: Optional[List[int]] = None
) -> None:
    """Plot the same metric from multiple training runs on the same axes."""
    # Set up the figure
    plt.figure(figsize=(10, 6), dpi=100)
    
    # Set font sizes
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size, labelsize=font_size)
    plt.rc('xtick', labelsize=font_size-1)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('legend', fontsize=font_size-1)
    
    # Default line styles and colors if not provided
    line_styles = line_styles or ['-', '--', '-.', ':']
    colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot each history
    for i, (history, label) in enumerate(zip(histories, labels)):
        train_metrics = []
        val_metrics = []
        
        for epoch in history:
            # Get training metric if it exists
            if 'train' in epoch and metric in epoch['train']:
                train_metrics.append(epoch['train'][metric])
            
            # Get validation metric if it exists
            if 'valid' in epoch and metric in epoch['valid']:
                val_metrics.append(epoch['valid'].get(metric, None))
        
        if not train_metrics:
            print(f"Warning: Metric '{metric}' not found in training data for {label}")
            continue
        
        # Apply epoch limit if specified
        max_epoch = epoch_limits[i] if epoch_limits and i < len(epoch_limits) else None
        
        # Prepare training data
        epochs = range(1, len(train_metrics) + 1)
        
        # Apply epoch limit by taking first N elements
        if max_epoch is not None:
            plot_epochs = list(epochs)[:max_epoch]
            plot_train = train_metrics[:max_epoch]
            plot_val = val_metrics[:max_epoch] if val_metrics else []
        else:
            plot_epochs = list(epochs)
            plot_train = train_metrics
            plot_val = val_metrics if val_metrics else []
        
        line_style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        
        # Plot training data
        plt.plot(plot_epochs, plot_train, 
                linestyle=line_style, 
                color=color,
                linewidth=2,
                label=f"{label}")
        
        # Plot validation data if available
        if plot_val and any(v is not None for v in plot_val):
            plt.plot(plot_epochs[:len(plot_val)], 
                    [v for v in plot_val if v is not None],
                    linestyle=line_style,
                    color=color,
                    linewidth=2,
                    alpha=0.7,
                    label=f"{label} (Val)")
    
    # Format title with Unicode subscript for A_{100}
    title_text = title or f'Training and Validation {metric}'
    
    plt.title(title_text, fontsize=font_size+2, pad=20)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel or metric, fontsize=font_size)
    
    # Customize grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Create output filename
    filename = f"{output_name}.png" if output_name else f"combined_{metric}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Plot saved to: {os.path.abspath(output_path)}")

def main():
    args = parse_arguments()
    
    try:
        # Load all histories
        histories = [load_history(path) for path in args.json_paths]
        
        # Generate default labels if not provided
        if args.legend and len(args.legend) == len(args.json_paths):
            labels = args.legend
        else:
            labels = [f"Run {i+1}" for i in range(len(args.json_paths))]
        
        # Set output directory
        output_dir = args.output if args.output else "outputs"
        
        # Generate the combined plot
        plot_metrics(
            histories=histories,
            labels=labels,
            metric=args.metric,
            output_dir=output_dir,
            output_name=args.name,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            font_size=args.font_size,
            line_styles=args.line_styles,
            colors=args.colors,
            epoch_limits=args.epoch_limits
        )
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
    except json.JSONDecodeError:
        print(f"Error: One or more files are not valid JSON")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
