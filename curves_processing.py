import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MaxNLocator

SAVE_DIR = "/mnt/saves"
TEMP_DIR = "/mnt/temp2"

ACADEMIC_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 12, 
    "font.size": 12, 
    "legend.fontsize": 11,
    "xtick.labelsize": 11, 
    "ytick.labelsize": 11
}

def get_resampled_cluster_curves(entropy_data, answers, complementary=False):
    """
    Extracts and resamples conditional entropy curves for a specific cluster of answers.
    Uses CubicSpline interpolation to map curves to a common normalized grid.

    Args:
        entropy_data (dict): The dictionary containing entropy curve results.
        answers (str or list): The target answer(s) to filter the cluster.
        complementary (bool, optional): If True, returns curves for answers NOT in `answers`. Defaults to False.

    Returns:
        np.ndarray: A 2D numpy array containing the resampled curves. Returns an empty array if no valid curves exist.
    """
    if isinstance(answers, str):
        answers = [answers]

    cluster_curves = [
        item["curve"] for item in entropy_data.get("results", [])
        if (item.get("original_answer") in answers) is not complementary
    ]

    non_empty_curves = [c for c in cluster_curves if len(c) > 0]
    if not non_empty_curves:
        return np.array([])
    
    avg_len = int(round(np.mean([len(c) for c in non_empty_curves])))
    if avg_len < 2:
        return np.array([])

    x_common_grid = np.linspace(0, 1, avg_len)
    resampled_curves = []
    
    for curve in cluster_curves:
        if len(curve) < 2:
            continue
        x_normalized = np.linspace(0, 1, len(curve))
        cs = CubicSpline(x_normalized, curve)
        resampled_curves.append(cs(x_common_grid))

    return np.array(resampled_curves)

def plot_points_and_visualize_mbe(plot_data, dest_path, ylabel="Conditional Entropy", xlabel="Reasoning Steps"):
    """
    Plots two comparative entropy curves and visualizes the Mean Between-group Error (MBE) 
    using shaded areas and vertical segments.

    Args:
        plot_data (list): A list containing tuples of (data, styling_info_dict) for the two curves.
        dest_path (str): The output file path for the generated plot image.
        ylabel (str, optional): Label for the y-axis. Defaults to "Conditional Entropy".
        xlabel (str, optional): Label for the x-axis. Defaults to "Reasoning Steps".
    """
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        if len(plot_data) < 2:
            print("Warning: At least two curves are required for MBE.")
            plt.close(fig)
            return

        ref_raw, ref_info = plot_data[1]
        comp_raw, comp_info = plot_data[0]
        
        y_ref = np.array(ref_raw[0]) if isinstance(ref_raw, tuple) else np.array(ref_raw)
        y_comp = np.array(comp_raw[0]) if isinstance(comp_raw, tuple) else np.array(comp_raw)
        
        x_ref = np.arange(1, 1 + len(y_ref))
        x_comp = np.arange(1, 1 + len(y_comp))

        min_len = min(len(y_ref), len(y_comp))
        steps = np.arange(1, min_len + 1)
        diffs = y_ref[:min_len] - y_comp[:min_len]

        ax.plot(x_ref, y_ref, "o-", color=ref_info.get("color", "blue"), label=ref_info.get("desc", "Ref"), linewidth=2, markersize=6, zorder=5)
        ax.plot(x_comp, y_comp, "o-", color=comp_info.get("color", "red"), label=comp_info.get("desc", "Comp"), linewidth=2, markersize=6, zorder=4)

        for x, y1, y2, d in zip(steps, y_ref[:min_len], y_comp[:min_len], diffs):
            ax.vlines(x, y1, y2, color="green" if d >= 0 else "red", linewidth=2.5, alpha=0.8, zorder=3)

        ax.fill_between(steps, y_ref[:min_len], y_comp[:min_len], where=(diffs >= 0), color="green", alpha=0.15, interpolate=True, zorder=2)
        ax.fill_between(steps, y_ref[:min_len], y_comp[:min_len], where=(diffs < 0), color="red", alpha=0.15, interpolate=True, zorder=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(frameon=True, loc="best")
        ax.set_xticks(steps)
        
        plt.tight_layout()
        plt.savefig(dest_path, dpi=300)
        plt.close()

def plot_generic_curves(plot_data, dest_path, ylabel="Value", xlabel="Steps", x_start=1):
    """
    Plots a generic set of curves, optionally including standard deviation uncertainty bands.

    Args:
        plot_data (list): A list containing tuples of (data, styling_info_dict). Data can be a 1D array or a (mean, std) tuple.
        dest_path (str): The output file path for the generated plot image.
        ylabel (str, optional): Label for the y-axis. Defaults to "Value".
        xlabel (str, optional): Label for the x-axis. Defaults to "Steps".
        x_start (int, optional): The starting value for the x-axis. Defaults to 1.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        legend_added = set()

        for data, info in plot_data:
            if isinstance(data, tuple) and len(data) == 2:
                y_mean, y_std = np.array(data[0]), np.array(data[1])
            else:
                y_mean, y_std = np.array(data), None

            if y_mean.size == 0: 
                continue

            x_axis = np.arange(x_start, x_start + len(y_mean))
            label = info.get("desc", "")
            plot_label = label if label and label not in legend_added else None
            if plot_label: 
                legend_added.add(plot_label)

            color = info.get("color", "blue")
            ax.plot(x_axis, y_mean, color=color, linestyle=info.get("linestyle", "-"), label=plot_label, marker="o", markersize=6, linewidth=2)

            if y_std is not None:
                ax.fill_between(x_axis, y_mean - y_std, y_mean + y_std, color=color, alpha=0.15, linewidth=0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
        
        if legend_added: 
            ax.legend(frameon=True, loc="best")

        plt.tight_layout()
        plt.savefig(dest_path, dpi=300)
        plt.close()

def load_entropy_data_from_json(filename):
    """
    Loads entropy data from a JSON file and converts curve lists back into numpy arrays.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        dict: The loaded data dictionary with curves formatted as np.ndarray.
    """
    with open(filename, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    for item in loaded_data["results"]:
        item["curve"] = np.array(item["curve"])
    
    print(f"Data loaded from {filename} and curves converted to np.array.")
    return loaded_data

def main():
    """
    Main entry point. Loads the entropy data for various math problems, 
    processes the curves for correct and wrong answers, and generates the resulting plots.
    """
    problems = [f"int_algebra_lv4_{i+1}" for i in range(4)] + [f"int_algebra_lv5_{i+1}" for i in range(5)]

    # Analysis 1
    for problem in problems:
        for i in range(3):
            try:
                entropy_data = load_entropy_data_from_json(f"{SAVE_DIR}/entropy_curves_{problem}_answer_{i}.json")
                with open(f"{SAVE_DIR}/cots_{problem}.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                continue
            
            correct_resampled_curves = get_resampled_cluster_curves(entropy_data, entropy_data["target_answer"])
            wrong_resampled_curves = get_resampled_cluster_curves(entropy_data, entropy_data["target_answer"], complementary=True)
            
            if len(correct_resampled_curves) == 0 or len(wrong_resampled_curves) == 0:
                continue
            
            average_correct = correct_resampled_curves.mean(axis=0)
            margin_correct = 2 * (correct_resampled_curves.std(axis=0, ddof=1) / np.sqrt(correct_resampled_curves.shape[0]))

            average_wrong = wrong_resampled_curves.mean(axis=0)
            margin_wrong = 2 * (wrong_resampled_curves.std(axis=0, ddof=1) / np.sqrt(wrong_resampled_curves.shape[0]))
            
            class_avg_correct = {"desc": r"Cluster $Y_i$", "color": "blue", "alpha": 1.0}
            class_avg_wrong = {"desc": r"Cluster $\neg Y_i$", "color": "red", "alpha": 1.0}
            
            plot_data_mbe = [
                (average_correct, class_avg_correct),
                (average_wrong, class_avg_wrong)
            ]
            plot_points_and_visualize_mbe(plot_data_mbe, dest_path=f"{TEMP_DIR}/{problem}_answer_{i}_mbe_plot.png")

            plot_data_curves = [
                ((average_correct, margin_correct), class_avg_correct),
                ((average_wrong, margin_wrong), class_avg_wrong)
            ]
            plot_generic_curves(plot_data_curves, dest_path=f"{TEMP_DIR}/{problem}_answer_{i}_plot.png")

    # Analysis 2
    for problem in problems:
        try:
            entropy_data = load_entropy_data_from_json(f"{SAVE_DIR}/entropy_curves_{problem}_answer_0.json")
        except FileNotFoundError:
            continue
        
        curves = [cot["curve"] for cot in entropy_data["results"] if cot["original_answer"] == entropy_data["target_answer"]]
        
        cot_idx = 0
        if len(curves) > cot_idx:
            cot = curves[cot_idx]
            cot_class = {"desc": f"{entropy_data['target_answer']}", "color": "blue"}
            
            plot_curves = [(cot, cot_class)]
            plot_generic_curves(plot_curves, f"{TEMP_DIR}/{problem}_wrong_1_cot_{cot_idx}_plot.png", "Conditional Entropy", "Reasoning Step", 1)

if __name__ == "__main__":
    main()