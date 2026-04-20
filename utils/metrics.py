import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
import os
from scipy.ndimage import gaussian_filter


import pandas as pd

def alpha_plots():
    df = pd.read_csv("data/rgd_analysis.csv")

    metrics = ["Gradient Average", "Saturation Average", "Gradient Max", "Intensity Average"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    ahat = r"$\hat{\alpha}$"

    for ax, metric in zip(axes, metrics):
        ax.plot(df["Alpha"], df[metric], marker="o", linewidth=2, markersize=5)
        ax.set_xlabel(ahat)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}")
        ax.grid(True, alpha=0.3)

    # plt.suptitle("Metrics vs Alpha", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots.png", dpi=150, bbox_inches="tight")
    plt.show()



def load_data(LAB_file, RGB_file):
    lab_vals = []
    with open(LAB_file) as f:
        for line in f:
            l, a, b = line.strip().split(",")
            lab_vals.append([float(l), float(a), float(b)])

    lab_vals = np.array(lab_vals)

    lookup = np.zeros((256, 256, 256, 3), dtype=np.float32)

    with open(RGB_file) as f:
        for i, line in enumerate(f):
            r, g, b = map(int, line.strip().split(","))
            lookup[r, g, b] = lab_vals[i]
    return lookup

def get_gradient(lookup_data):
    grad_x, grad_y, grad_z = np.gradient(lookup_data, axis=(0,1,2))
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    magnitude = np.sum(grad)
    print("Gradient max: " + str(np.max(grad)))
    print("Gradient magntidue (average): " + str(magnitude/16777216.0))

def get_histogram(lookup_data: np.ndarray, save_path: str, title: str = "Hue Distribution", bins: int = 360):
    """
    Plot a hue histogram for a (256,256,256,3) RGB color cube and save it.
   
    Args:
        cube:      np.ndarray of shape (..., 3), values in [0, 255] or [0.0, 1.0]
        save_path: path to save the figure, e.g. "output/cube1_hue.png"
        title:     plot title
        bins:      number of hue bins (360 = 1° per bin)
    """
    # --- Normalize to [0, 1] if needed ---
    rgb = lookup_data.reshape(-1, 3).astype(np.float32)
    # print(rgb.max())
    # print(rgb.min())    
    if rgb.max() > 1.0:
        rgb /= 255.0

    # print(rgb.max())
    # print(rgb.min())
    # --- RGB → HSV, extract hue ---
    # matplotlib's rgb_to_hsv expects shape (..., 3)
    hsv = rgb_to_hsv(rgb)        # shape (N, 3)
    hue = hsv[:, 0]              # values in [0, 1], where 1.0 == 360°

    # --- Mask out near-achromatic colors (low saturation) ---
    # Optional but recommended — greys have meaningless/noisy hue values
    # saturation = hsv[:, 1]
    # value      = hsv[:, 2]
    # mask = (saturation > 0.1) & (value > 0.05)
    # hue = hue[mask]

    # --- Build histogram ---
    counts, edges = np.histogram(hue, bins=bins, range=(0, 1))

    # --- Color each bar by its hue ---
    bin_centers = (edges[:-1] + edges[1:]) / 2          # shape (bins,)
    colors = hsv_to_rgb(
        np.stack([bin_centers,
                  np.ones(bins),
                  np.ones(bins)], axis=-1)               # full sat + value
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(bin_centers * 360, counts, width=360 / bins,
           color=colors, edgecolor='none')

    ax.set_xlabel("Hue (degrees)", fontsize=12)
    ax.set_ylabel("Pixel count", fontsize=12)
    # ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°\nRed", "60°\nYellow", "120°\nGreen",
                         "180°\nCyan", "240°\nBlue", "300°\nMagenta", "360°"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def get_richness(lookup_data: np.ndarray, sample_size: int = 50_000):
    """
    Compute richness metrics across hue, saturation, and intensity (value).

    Returns a dict of scalar metrics — higher generally means richer/more diverse.
    """
    # --- Flatten + normalize ---
    pixels = lookup_data.reshape(-1, 3).astype(np.float32)
    if pixels.max() > 1.0:
        pixels /= 255.0

    # --- Subsample ---
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), size=sample_size, replace=False)
        pixels = pixels[idx]

    # --- Convert to HSV ---
    hsv    = rgb_to_hsv(pixels)          # shape (N, 3)
    hue    = hsv[:, 0]                   # [0, 1]
    sat    = hsv[:, 1]                   # [0, 1]
    val    = hsv[:, 2]                   # [0, 1] (intensity/brightness)

    print("Saturation mean: " + str(sat.mean()))
    print("Intensity mean: " + str(val.mean()))

def hue_histogram_compare_2(
    cube_a: np.ndarray, label_a: str,
    cube_b: np.ndarray, label_b: str,
    save_path: str,
    bins: int = 360,
    sample_size: int = 50_000,
):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    datasets = [(cube_a, label_a, axes[0]), (cube_b, label_b, axes[1])]

    for cube, label, ax in datasets:
        # --- Flatten + normalize ---
        pixels = cube.reshape(-1, 3).astype(np.float32)
        if pixels.max() > 1.0:
            pixels /= 255.0

        # --- Subsample ---
        if len(pixels) > sample_size:
            idx = np.random.choice(len(pixels), size=sample_size, replace=False)
            pixels = pixels[idx]

        # --- HSV + mask achromatic ---
        hsv = rgb_to_hsv(pixels)
        hue = hsv[:, 0]
        sat = hsv[:, 1]
        val = hsv[:, 2]
        mask = (sat > 0.1) & (val > 0.05)
        hue = hue[mask]

        # --- Normalize to density so different pixel counts are comparable ---
        counts, edges = np.histogram(hue, bins=bins, range=(0, 1))
        density = counts / counts.sum()  # fraction of chromatic pixels per bin

        # --- Color each bar by its hue ---
        bin_centers = (edges[:-1] + edges[1:]) / 2
        colors = hsv_to_rgb(
            np.stack([bin_centers, np.ones(bins), np.ones(bins)], axis=-1)
        )

        ax.bar(bin_centers * 360, density * 100,  # convert to %
               width=360 / bins, color=colors, edgecolor='none')

        # --- Label with pixel count so reader knows the sample sizes ---
        total_px = cube.reshape(-1, 3).shape[0]
        ax.set_title(f"{label}  ({total_px:,} pixels)", fontsize=12, fontweight='bold')
        ax.set_ylabel("% of chromatic pixels", fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, None)

    axes[1].set_xlabel("Hue (degrees)", fontsize=12)
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks([0, 60, 120, 180, 240, 300, 360])
    axes[1].set_xticklabels(["0°\nRed", "60°\nYellow", "120°\nGreen",
                              "180°\nCyan", "240°\nBlue", "300°\nMagenta", "360°"])

    plt.suptitle("Hue Distribution Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(save_path)}")

def hue_histogram_compare(
    cube_a: np.ndarray, label_a: str,
    cube_b: np.ndarray, label_b: str,
    save_path: str,
    bins: int = 360,
    sample_size: int = 50_000,
):
    def get_density(cube):
        pixels = cube.reshape(-1, 3).astype(np.float32)
        if pixels.max() > 1.0:
            pixels /= 255.0
        if len(pixels) > sample_size:
            idx = np.random.choice(len(pixels), size=sample_size, replace=False)
            pixels = pixels[idx]
        hsv = rgb_to_hsv(pixels)
        mask = (hsv[:, 1] > 0.1) & (hsv[:, 2] > 0.05)
        counts, edges = np.histogram(hsv[mask, 0], bins=bins, range=(0, 1))
        return counts / counts.sum() * 100, edges

    density_a, edges = get_density(cube_a)
    density_b, _      = get_density(cube_b)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    colors = hsv_to_rgb(np.stack([bin_centers, np.ones(bins), np.ones(bins)], axis=-1))

    # Shared y max so both plots are directly comparable
    y_max = max(density_a.max(), density_b.max()) * 1.1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)

    for ax, density, cube, label in [
        (ax1, density_a, cube_a, label_a),
        (ax2, density_b, cube_b, label_b),
    ]:
        ax.bar(bin_centers * 360, density, width=360 / bins,
               color=colors, edgecolor='none')
        total_px = cube.reshape(-1, 3).shape[0]
        # ax.set_title(f"{label}  ({total_px:,} pixels)", fontsize=12, fontweight='bold')
        ax.set_ylabel("% of chromatic pixels", fontsize=10)
        ax.set_ylim(0, y_max)
        ax.grid(True, axis='y', alpha=0.3)

    ax2.set_xlabel("Hue (degrees)", fontsize=12)
    ax2.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax2.set_xticklabels(["0°\nRed", "60°\nYellow", "120°\nGreen",
                          "180°\nCyan", "240°\nBlue", "300°\nMagenta", "360°"])

    # plt.suptitle("Hue Distribution Comparison", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {os.path.abspath(save_path)}")

def run_metrics():
    original_filepath = "data/OriginalLABVals.txt"
    new_filepath = "AllCandidateLABvals_CIELAB_1_RGD_50_neural_256.txt"
    rgb_filepath = "data/AllCorrespondingRGBVals.txt"

    # hue_histogram_compare(load_data(original_filepath, rgb_filepath), "original", load_data(new_filepath, rgb_filepath), "new", "results/histograms/hue_comparison.png")
    filepaths = [original_filepath, # original baseline, 1
                #  "AllCandidateLABvals_CIELAB_1_RGD_05_neural_256.txt", # neural bounding tests, various alpha values, 3
                #  "AllCandidateLABvals_CIELAB_1_RGD_15_neural_256.txt", 
                #  "AllCandidateLABvals_CIELAB_1_RGD_25_neural_256.txt", 
                #  "AllCandidateLABvals_CIELAB_1_RGD_35_neural_256.txt", 
                #  "AllCandidateLABvals_CIELAB_1_RGD_50_neural_256.txt",
                 "AllCandidateLABvals_CIELAB_1_RGD_75_neural_256.txt"]
                #  "AllCandidateLABvals_CIELAB_1_RGD_100_neural_256.txt",
                #  "AllCandidateLABvals_CIELAB_1_RGD_125_neural_256.txt",
                #  "AllCandidateLABvals_CIELAB_1_RGD_150_neural_256.txt"]
                #  "AllCandidateLABvals_CIELAB_1_Euclidean.txt", # versus euclidean, cielab (neural same)
                #  "AllCandidateLABvals_CIELAB_1_RGD_05.txt", # 4
                #  "AllCandidateLABvals_CIELAB_1_RGD_05_sigma_0o25_vox_256.txt", # versus gaussian filtering
                #  "AllCandidateLABvals_CIELAB_1_RGD_05_sigma_2o0_vox_256.txt", 
                #  "AllCandidateLABvals_CIELAB_1_RGD_05_sigma_4o0_vox_256.txt", # 2
                #  "AllCandidateLABvals_OKLAB_1_RGD_05_sigma_4o0_vox_256.txt", # versus oklab 5
                #  "AllCandidateLABvals_OKLAB_1_RGD_25_sigma_4o0_vox_256.txt", 
                #  "AllCandidateLABvals_RGB_1_euclidean.txt"] # versus rgb euclidean 
    
    # lookup_data = load_data(original_filepath, rgb_filepath)

    # filtered_data = gaussian_filter(lookup_data, sigma=32.0, axes=(0,1,2))
    # new_LAB_file = open("Smoothed_OriginalLABVals_64.txt", "w")
    # for r in range(0, 256):
    #     for g in range(0, 256):
    #         for b in range(0, 256):
    #             interpolatedLAB = filtered_data[r][g][b]
    #             new_LAB_file.write(str(float(interpolatedLAB[0])) + "," + str(float(interpolatedLAB[1])) + "," + str(float(interpolatedLAB[2])) + "\n")
    # print("saved to " + "Smoothed_OriginalLABVals.txt")

    alpha_plots()
    
    # for filepath in filepaths:
    #     lookup_data = load_data(filepath, rgb_filepath)
    
    #     print("File: " + filepath)
    #     get_gradient(lookup_data)
    #     name = filepath.split(".")[0]
    #     if name.count("/") > 0:
    #         name = name.split("/")[1]
    #     get_histogram(lookup_data, save_path=f"results/histograms/{name}_hue.png", title=f"Hue Distribution — {name}")
    #     get_richness(lookup_data)

    #     print(" ")
    #     print("__________________________")
    #     print(" ")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# -----------------------------------------------------------------------
# Load and filter
# -----------------------------------------------------------------------

def load_and_filter(csv_path):
    df = pd.read_csv(csv_path)

    # Sort by frame number — out-of-order writes shouldn't happen anymore
    # with synchronous readback, but sort anyway to be safe
    df = df.sort_values("frame").reset_index(drop=True)

    # Remove duplicate frame entries
    df = df.drop_duplicates(subset="frame", keep="first").reset_index(drop=True)

    # Filter lagged rows: flag anything where the frame jump is more than
    # 10x the median step size
    frame_diffs = df["frame"].diff()
    median_step = frame_diffs.median()
    lag_threshold = median_step * 10
    lagged_mask = (frame_diffs > lag_threshold) & (frame_diffs.notna())

    n_lagged = lagged_mask.sum()
    if n_lagged > 0:
        print(f"Filtered out {n_lagged} lagged rows (frame jump > {lag_threshold:.0f} frames)")
        df = df[~lagged_mask].reset_index(drop=True)

    print(f"Loaded {len(df)} frames after filtering")
    print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
    print(f"Time range:  {df['time_seconds'].min():.3f}s to {df['time_seconds'].max():.3f}s")

    return df


# -----------------------------------------------------------------------
# Compute gradients
# -----------------------------------------------------------------------

# def compute_gradients(df):
#     """
#     Computes per-frame gradient (rate of change) for each region:
#       - label:      average color of the video pixels under the label mask
#       - background: average color of the video pixels under the background mask
#       - rendered:   actual color the shader displayed on the label (from lookup table)

#     Gradient magnitude = Euclidean distance in RGB space between consecutive frames.
#     """
#     results = {}

#     regions = {
#         "label":      ("label_r",      "label_g",      "label_b"),
#         "background": ("background_r", "background_g", "background_b"),
#         "rendered":   ("rendered_r",   "rendered_g",   "rendered_b"),
#     }

#     for region, (r_col, g_col, b_col) in regions.items():
#         dr = df[r_col].diff()
#         dg = df[g_col].diff()
#         db = df[b_col].diff()

#         magnitude = np.sqrt(dr**2 + dg**2 + db**2)

#         results[region] = {
#             "magnitude": magnitude,
#             "dr": dr,
#             "dg": dg,
#             "db": db,
#         }

#     return results

def compute_gradients(df):
    dt = df["frame"].diff()  # actual time gap between rows in seconds

    results = {}

    regions = {
        "label":      ("label_r",      "label_g",      "label_b"),
        "background": ("background_r", "background_g", "background_b"),
        "rendered":   ("rendered_r",   "rendered_g",   "rendered_b"),
    }

    for region, (r_col, g_col, b_col) in regions.items():
        dr = df[r_col].diff() 
        dg = df[g_col].diff() 
        db = df[b_col].diff() 

        magnitude = np.sqrt(dr**2 + dg**2 + db**2)

        results[region] = {
            "magnitude": magnitude,
            "dr": dr,
            "dg": dg,
            "db": db,
        }

    return results

# -----------------------------------------------------------------------
# Stats
# -----------------------------------------------------------------------

def print_stats(df, gradients):
    print("\n--- Gradient Statistics ---")
    for region, data in gradients.items():
        mag = data["magnitude"].dropna()
        max_frame = mag.idxmax()
        max_time  = df.loc[max_frame, "time_seconds"] if max_frame in df.index else "?"
        print(f"\n{region.upper()} color gradient (RGB Euclidean distance per frame):")
        print(f"  Max gradient:     {mag.max():.4f}  (frame {df.loc[max_frame, 'frame']}, t={max_time:.3f}s)")
        print(f"  Average gradient: {mag.mean():.4f}")
        print(f"  Std deviation:    {mag.std():.4f}")
        print(f"  Median gradient:  {mag.median():.4f}")

    return {
        region: {
            "max":       data["magnitude"].dropna().max(),
            "avg":       data["magnitude"].dropna().mean(),
            "max_frame": data["magnitude"].dropna().idxmax(),
        }
        for region, data in gradients.items()
    }


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot(df, gradients, output_path="color_gradients.png"):
    time = df["time_seconds"]

    # 4 rows: raw RGB for each of the 3 regions + magnitude comparison
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    channel_colors = {"r": "#ff4444", "g": "#44ff88", "b": "#4488ff"}

    region_info = {
        "label":      {"title": "Label Region (background under mask)", "accent": "#ff8844"},
        "background": {"title": "Background Region",                     "accent": "#44aaff"},
        "rendered":   {"title": "Rendered Label Color (from LUT)",       "accent": "#cc44ff"},
    }

    col_map = {
        "label":      ("label_r",      "label_g",      "label_b"),
        "background": ("background_r", "background_g", "background_b"),
        "rendered":   ("rendered_r",   "rendered_g",   "rendered_b"),
    }

    # ---- Rows 0-2: Raw RGB values per region ----
    for row_idx, (region, info) in enumerate(region_info.items()):
        ax = fig.add_subplot(gs[row_idx, :2])
        ax.set_facecolor("#1a1a1a")
        r_col, g_col, b_col = col_map[region]
        ax.plot(time, df[r_col], color=channel_colors["r"], linewidth=0.8, alpha=0.9, label="R")
        ax.plot(time, df[g_col], color=channel_colors["g"], linewidth=0.8, alpha=0.9, label="G")
        ax.plot(time, df[b_col], color=channel_colors["b"], linewidth=0.8, alpha=0.9, label="B")
        ax.set_title(f"{info['title']} — Raw RGB", color="white", fontsize=10)
        ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Value (0–255)", color="#aaaaaa", fontsize=8)
        ax.set_ylim(0, 255)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#333333")
        ax.legend(fontsize=8, labelcolor="white", facecolor="#2a2a2a", edgecolor="#444444")

        # Per-channel gradient in right column
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.set_facecolor("#1a1a1a")
        ax2.plot(time, gradients[region]["dr"].abs(), color=channel_colors["r"], linewidth=0.7, alpha=0.9, label="|ΔR|")
        ax2.plot(time, gradients[region]["dg"].abs(), color=channel_colors["g"], linewidth=0.7, alpha=0.9, label="|ΔG|")
        ax2.plot(time, gradients[region]["db"].abs(), color=channel_colors["b"], linewidth=0.7, alpha=0.9, label="|ΔB|")
        ax2.set_title(f"{info['title']} — Per-Channel Gradient", color="white", fontsize=10)
        ax2.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
        ax2.set_ylabel("|ΔValue| per frame", color="#aaaaaa", fontsize=8)
        ax2.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax2.spines.values(): spine.set_edgecolor("#333333")
        ax2.legend(fontsize=8, labelcolor="white", facecolor="#2a2a2a", edgecolor="#444444")

    # ---- Row 3: Magnitude comparison across all three regions ----
    ax3 = fig.add_subplot(gs[3, :])
    ax3.set_facecolor("#1a1a1a")
    for region, info in region_info.items():
        mag = gradients[region]["magnitude"]
        avg = mag.dropna().mean()
        mx  = mag.dropna().max()
        ax3.plot(time, mag, linewidth=0.9, color=info["accent"], alpha=0.85, label=f"{region} (avg={avg:.2f}, max={mx:.2f})")
        ax3.axhline(avg, color=info["accent"], linewidth=0.8, linestyle="--", alpha=0.4)

    ax3.set_title("RGB Change Magnitude — All Regions", color="white", fontsize=11)
    ax3.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
    ax3.set_ylabel("Euclidean distance per frame", color="#aaaaaa", fontsize=9)
    ax3.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor("#333333")
    ax3.legend(fontsize=9, labelcolor="white", facecolor="#2a2a2a", edgecolor="#444444")

    fig.suptitle("Label Color Gradient Analysis", color="white", fontsize=14, y=0.99)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nPlot saved to: {output_path}")
    plt.show()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def process_final_colors(csv_filepath, output_path):
    # CSV_PATH    = "label_colors_export.csv"   # update path if needed
    # OUTPUT_PATH = "color_gradients.png"

    df        = load_and_filter(csv_filepath)
    gradients = compute_gradients(df)
    stats     = print_stats(df, gradients)
    plot(df, gradients, output_path)