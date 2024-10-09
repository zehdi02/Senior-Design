import os
import pandas as pd
import matplotlib.pyplot as plt

# aggregate csv of training results from multiple runs of YOLO model
def aggregate_run_results():
    # Path to the directory containing the training results
    results_dir = "runs/detect/"
    results = []

    # Iterate through each run directory
    for run in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, run)
        if os.path.isdir(run_dir):
            # Load the training results CSV file
            results_file = os.path.join(run_dir, "results.csv")
            if os.path.exists(results_file):
                run_results = pd.read_csv(results_file)
                run_results.columns = run_results.columns.str.strip()
                run_results["run"] = run
                if len(results) > 0:
                    run_results["epoch"] += results[-1]["epoch"].max()
                results.append(run_results)

    # Concatenate all results into a single DataFrame
    all_results = pd.concat(results)
    results.to_csv('results.csv', index=False)

    plot_results(all_results)

    return all_results

def plot_results(results):
    metrics = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
        'metrics/precision(B)', 'metrics/recall(B)',
        'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]

    plt.figure(figsize=(24, 12))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 5, i)
        plt.plot(results['epoch'], results[metric], 'o', label=metric)  # Dots for each point
        plt.plot(results['epoch'], results[metric], 'o--', label=f'{metric} (smoothed)')  # Dotted line for smooth curve
        plt.legend()
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')

    # Save the plot as a 2400x1200 image
    plt.tight_layout()
    plt.savefig('results.png', dpi=100)
    plt.show()


def main():
    aggregate_run_results()

if __name__ == '__main__':
    main()