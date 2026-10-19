
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

def generate_metrics_plots(metrics_file: Path, output_dir: Path):
    """
    Reads metrics_record.json and generates visualization plots for BRI, NPS, SoV, and Polarization.
    """
    if not metrics_file.exists():
        print(f"[Visualization] Metrics file not found: {metrics_file}")
        return

    try:
        data = json.loads(metrics_file.read_text(encoding='utf-8'))
        round_metrics = data.get("round_metrics", [])
        
        if not round_metrics:
            print("[Visualization] No round metrics found.")
            return

        timesteps = [m.get("timestep", 0) for m in round_metrics]
        bri = [m.get("market_metrics", {}).get("BRI", 0.0) for m in round_metrics]
        # Round metrics already use Appendix C units: BRI/NPS in points, SoV/Polarization in percent.
        nps = [m.get("market_metrics", {}).get("NPS", 0.0) for m in round_metrics]
        sov = [m.get("market_metrics", {}).get("SoV", 0.0) or m.get("market_metrics", {}).get("Brand_Current_SoV", 0.0) for m in round_metrics]
        polarization = [m.get("market_metrics", {}).get("PolarizationIndex", 0.0) for m in round_metrics]

        # 1. Combined Metrics Plot (BRI, NPS, SoV, Polarization)
        plt.figure(figsize=(14, 8))
        
        # Create a primary y-axis for BRI (usually negative or small range)
        ax1 = plt.gca()
        l1, = ax1.plot(timesteps, bri, label="BRI (Reputation)", marker="o", linestyle="-", color="tab:blue", linewidth=2)
        ax1.set_xlabel("Timestep", fontsize=12)
        ax1.set_ylabel("Brand Reputation Index (BRI)", color="tab:blue", fontsize=12)
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        # Create a secondary y-axis for SoV and Polarization (0-1 range)
        ax2 = ax1.twinx()
        l2 = None
        l3 = None
        l4 = None
        
        if sov and any(sov):
            l2, = ax2.plot(timesteps, sov, label="SoV (Share of Voice)", marker="s", linestyle="--", color="tab:green", linewidth=2)
        
        if polarization and any(polarization):
            l3, = ax2.plot(timesteps, polarization, label="Polarization Index", marker="^", linestyle="-.", color="tab:purple", linewidth=2)
            
        ax2.set_ylabel("SoV / Polarization (%)", color="black", fontsize=12)
        ax2.set_ylim(0, 100)

        # Create a third y-axis for NPS (-100 to 100 range)
        ax3 = ax1.twinx()
        # Offset the right spine of ax3.  The ticks and label have already been
        # placed on the right by twinx above.
        ax3.spines["right"].set_position(("axes", 1.1))
        # Having been created by twinx, ax3 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        for sp in ax3.spines.values():
            sp.set_visible(False)
        # Second, show the right spine.
        ax3.spines["right"].set_visible(True)
        
        if nps and any(nps):
             l4, = ax3.plot(timesteps, nps, label="NPS (Net Promoter Score)", marker="*", linestyle=":", color="tab:red", linewidth=2)
        
        ax3.set_ylabel("NPS (-100 to 100)", color="tab:red", fontsize=12)
        ax3.tick_params(axis='y', labelcolor="tab:red")
        ax3.set_ylim(-100, 100)
        
        # Combine legends
        lines = [l for l in [l1, l2, l3, l4] if l is not None]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
        
        plt.title("Market Dynamics: Reputation, NPS, Voice & Polarization", fontsize=14, pad=20)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, right=0.85) # Make room for legend and third axis
        plt.savefig(output_dir / "metrics_combined_trend.png", dpi=300)
        plt.close()

        print(f"[Visualization] Plots saved to {output_dir}")

    except Exception as e:
        print(f"[Visualization] Error generating plots: {e}")
