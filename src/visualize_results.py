# src/visualize_results.py
import os
import yaml
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

class CatRiskVisualizer:
    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}.")
            
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            
        self.mode = config_data["model_settings"]["resolution_mode"]
        self.data_dir = f"./data/{self.mode}"
        self.output_dir = f"./outputs/{self.mode}"
        
        # FIXED: Explicitly define all missing paths to prevent AttributeErrors
        self.exposure_path = "./data/portfolio_exposure.csv"
        self.financial_matrix_path = f"{self.data_dir}/portfolio_financial_matrix.parquet"
        
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def plot_exceedance_probability_curve(self):
        """
        1. STANDARD EP CURVE: Plots continuous financial tail risk.
        """
        print(f"--- Plotting Exceedance Probability (EP) Curve [{self.mode.upper()}] ---")
        ep_path = f"{self.data_dir}/exceedance_probability_curve.csv"
        if not os.path.exists(ep_path):
            print(f"Skipping: EP data not found at {ep_path}.")
            return
            
        df_ep = pd.read_csv(ep_path)
        plt.figure(figsize=(10, 6))
        
        plt.plot(
            df_ep['return_period_years'], 
            df_ep['insured_loss_threshold'] / 1e6, 
            color='#1f77b4', linewidth=2.5, label='Insured Loss Threshold'
        )
        
        benchmark_rps = [10, 20, 50, 100, 250, 500]
        for rp in benchmark_rps:
            if rp > df_ep['return_period_years'].max():
                continue
            idx = (df_ep['return_period_years'] - rp).abs().idxmin()
            row = df_ep.loc[idx]
            loss_m = row['insured_loss_threshold'] / 1e6
            
            plt.scatter(row['return_period_years'], loss_m, color='red', s=50, zorder=5)
            plt.annotate(
                f"1-in-{rp}yr\n${loss_m:.1f}M",
                (row['return_period_years'], loss_m),
                textcoords="offset points", xytext=(10,-5),
                ha='left', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, ec="gray")
            )

        plt.xscale('log')
        plt.xlim(1, 1000)
        plt.title(f"Portfolio Exceedance Probability (EP) Curve\nResolution Mode: [{self.mode.upper()}]", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Return Period (Years) [Log Scale]", fontsize=11)
        plt.ylabel("Annual Insured Loss Severity ($ Millions)", fontsize=11)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="upper left")
        
        plt.savefig(f"{self.output_dir}/exceedance_probability_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_portfolio_loss_composition(self):
        """
        2. LOSS COMPOSITION ANALYSIS: Aggregates total loss metrics by construction type safely.
        """
        print(" -> Computing loss composition via memory-safe PyArrow streaming...")
        
        df_exposure = pd.read_csv(self.exposure_path)
        df_exposure.columns = [c.lower().strip() for c in df_exposure.columns]
        
        construction_map = dict(zip(df_exposure['property_id'].astype(str), df_exposure['construction'].astype(str)))
        loss_by_construction = {}
        
        parquet_file = pq.ParquetFile(self.financial_matrix_path)
        
        for batch in parquet_file.iter_batches(batch_size=50000, columns=['property_id', 'ground_up_loss', 'insured_loss']):
            prop_ids = batch.column('property_id').to_pylist()
            ground_up = batch.column('ground_up_loss').to_numpy()
            insured = batch.column('insured_loss').to_numpy()
            
            for pid, gu, ins in zip(prop_ids, ground_up, insured):
                const_type = construction_map.get(str(pid), 'Unknown')
                if const_type not in loss_by_construction:
                    loss_by_construction[const_type] = {'ground_up_loss': 0.0, 'insured_loss': 0.0}
                    
                loss_by_construction[const_type]['ground_up_loss'] += gu
                loss_by_construction[const_type]['insured_loss'] += ins

        df_plot = pd.DataFrame.from_dict(loss_by_construction, orient='index').reset_index()
        df_plot.columns = ['construction', 'ground_up_loss', 'insured_loss']
        
        df_melted = df_plot.melt(id_vars='construction', value_vars=['ground_up_loss', 'insured_loss'],
                                 var_name='Loss Type', value_name='Loss Value')
        df_melted['Loss Value ($ Millions)'] = df_melted['Loss Value'] / 1_000_000
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x='construction', y='Loss Value ($ Millions)', hue='Loss Type', palette='viridis')
        plt.title('Portfolio Loss Composition by Construction Type')
        plt.xlabel('Construction Type')
        plt.ylabel('Total Accumulated Losses ($ Millions)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_plot_path = os.path.join(self.output_dir, 'loss_composition_by_construction.png')
        plt.savefig(output_plot_path, dpi=300)
        plt.close()
        print(f"SUCCESS: Saved loss composition plot to {output_plot_path}")

    def plot_model_resolution_comparison(self):
        """
        3. MODEL RISK PLOT: Overlays Spatial vs Individual EP Curves.
        """
        print("--- Compiling Spatial vs. Individual Model Variance Plot ---")
        spatial_path = "./data/spatial/exceedance_probability_curve.csv"
        individual_path = "./data/individual/exceedance_probability_curve.csv"
        
        if not (os.path.exists(spatial_path) and os.path.exists(individual_path)):
            print("Skipping comparison plot: Ensure both 'spatial' and 'individual' pipelines have been executed first.")
            return
            
        df_sp = pd.read_csv(spatial_path)
        df_ind = pd.read_csv(individual_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_sp['return_period_years'], df_sp['insured_loss_threshold'] / 1e6, 
                 label='Spatial Anchor Profile', color='#ff7f0e', lw=2, ls='--')
        plt.plot(df_ind['return_period_years'], df_ind['insured_loss_threshold'] / 1e6, 
                 label='Point-Intercept Track Profile', color='#1f77b4', lw=2.5)
        
        common_len = min(len(df_sp), len(df_ind))
        plt.fill_between(
            df_ind['return_period_years'].iloc[:common_len],
            df_sp['insured_loss_threshold'].iloc[:common_len] / 1e6,
            df_ind['insured_loss_threshold'].iloc[:common_len] / 1e6,
            color='purple', alpha=0.15, label='Resolution Uncertainty Space'
        )
        
        plt.xscale('log')
        plt.xlim(1, 1000)
        plt.title("Model Resolution Discrepancy & Tail Variance Analysis", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Return Period (Years) [Log Scale]", fontsize=11)
        plt.ylabel("Annual Insured Claims Complexity ($ Millions)", fontsize=11)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="upper left")
        
        plt.savefig(f"{self.output_dir}/model_resolution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_tail_risk_tvar(self):
        """
        4. CO-RISK ANALYSIS: Tail VaR (TVaR) vs standard Value at Risk (VaR).
        """
        print(f"--- Calculating VaR vs. TVaR Tail Metrics [{self.mode.upper()}] ---")
        ep_path = f"{self.data_dir}/exceedance_probability_curve.csv"
        if not os.path.exists(ep_path):
            return
            
        df_ep = pd.read_csv(ep_path)
        losses = df_ep['insured_loss_threshold'].values / 1e6
        probs = df_ep['exceedance_probability'].values
        
        target_rps = [20, 50, 100, 250, 500]
        var_metrics = []
        tvar_metrics = []
        
        for rp in target_rps:
            if rp > df_ep['return_period_years'].max():
                continue
            target_prob = 1.0 / rp
            idx = (df_ep['return_period_years'] - rp).abs().idxmin()
            
            var_metrics.append(losses[idx])
            tvar_metrics.append(losses[probs <= target_prob].mean())
            
        if not var_metrics:
            return

        x = np.arange(len(var_metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, var_metrics, width, label='Value at Risk (VaR Threshold)', color='#4682B4', edgecolor='black', alpha=0.9)
        plt.bar(x + width/2, tvar_metrics, width, label='Tail VaR (TVaR / Expected Shortfall)', color='#B22222', edgecolor='black', alpha=0.9)
        
        plt.title(f"Capital Adequacy Tail Metric Stress Breakdown [{self.mode.upper()}]", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Disaster Horizon Window Return Period", fontsize=11)
        plt.ylabel("Loss Capital Requirements ($ Millions)", fontsize=11)
        plt.xticks(x, [f"1-in-{rp} Year" for rp in target_rps[:len(var_metrics)]])
        plt.grid(True, axis='y', ls='--', alpha=0.5)
        plt.legend(loc="upper left")
        
        for i, tvar in enumerate(tvar_metrics):
            plt.annotate(f"${tvar:.1f}M", (i + width/2, tvar), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#8B0000')
            
        plt.savefig(f"{self.output_dir}/portfolio_tvar_stress_test.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_geographic_accumulation_risk(self):
        """
        5. SPATIAL GEOGRAPHIC ACCUMULATION: Memory-safe chunked mapping.
        """
        print(f"--- Plotting Geospatial Accumulation Map with Real Basemap [{self.mode.upper()}] ---")
        
        if not os.path.exists(self.financial_matrix_path) or not os.path.exists(self.exposure_path):
            print("Skipping: Required map tracking datasets are missing.")
            return
            
        df_exp = pd.read_csv(self.exposure_path)
        df_exp.columns = [c.lower().strip() for c in df_exp.columns]
        
        # DYNAMIC LOOKUP: Resolve structural TIV column naming variations
        tiv_candidates = [c for c in df_exp.columns if 'tiv' in c or 'value' in c or 'insured' in c]
        tiv_col = tiv_candidates[0] if tiv_candidates else 'total_insured_value'

        # FIXED: Stream chunked accumulation mapping instead of heavy full reads
        print(" -> Aggregating geographic claims layout via streaming vectors...")
        accumulated_losses = {}
        parquet_file = pq.ParquetFile(self.financial_matrix_path)
        
        for batch in parquet_file.iter_batches(batch_size=50000, columns=['property_id', 'insured_loss']):
            prop_ids = batch.column('property_id').to_pylist()
            insured_losses = batch.column('insured_loss').to_numpy()
            
            for pid, loss in zip(prop_ids, insured_losses):
                accumulated_losses[str(pid)] = accumulated_losses.get(str(pid), 0.0) + loss

        # Map streamed sums onto exposure properties safely
        df_exp['accumulated_insured_loss'] = df_exp['property_id'].astype(str).map(accumulated_losses).fillna(0.0)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        pad = 0.5
        min_lon, max_lon = df_exp['longitude'].min() - pad, df_exp['longitude'].max() + pad
        min_lat, max_lat = df_exp['latitude'].min() - pad, df_exp['latitude'].max() + pad
        
        try:
            import contextily as cx
            
            def latlon_to_mercator(df):
                r_major = 6378137.0
                x = r_major * np.radians(df['longitude'])
                y = r_major * np.log(np.tan(np.pi / 4.0 + np.radians(df['latitude']) / 2.0))
                return x, y
            
            mx, my = latlon_to_mercator(df_exp)
            
            dummy_df_min = pd.DataFrame({'longitude': [min_lon], 'latitude': [min_lat]})
            dummy_df_max = pd.DataFrame({'longitude': [max_lon], 'latitude': [max_lat]})
            x_min, y_min = latlon_to_mercator(dummy_df_min)
            x_max, y_max = latlon_to_mercator(dummy_df_max)
            
            scatter = ax.scatter(
                mx, my,
                c=df_exp['accumulated_insured_loss'] / 1e6,
                s=df_exp[tiv_col] / 1e5,
                cmap='YlOrRd', alpha=0.8, edgecolors='black', linewidths=0.6, zorder=2
            )
            
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zorder=1)
            ax.set_xlim(x_min.values[0], x_max.values[0])
            ax.set_ylim(y_min.values[0], y_max.values[0])
            
        except ImportError:
            print("Contextily library not found. Falling back to default canvas framework...")
            ax.set_facecolor('#e0f2fe')
            
            scatter = ax.scatter(
                df_exp['longitude'], df_exp['latitude'],
                c=df_exp['accumulated_insured_loss'] / 1e6,
                s=df_exp[tiv_col] / 1e5,
                cmap='YlOrRd', alpha=0.85, edgecolors='black', linewidths=0.7, zorder=3
            )
            
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            ax.grid(True, color='white', linestyle='--', alpha=0.6, zorder=1)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label('Accumulated Insured Portfolio Claims ($ Millions)', rotation=270, labelpad=15, fontsize=11, fontweight='semibold')
        
        ax.set_title(f"Geographic Portfolio Loss Concentration Map\nResolution Layer Mode: [{self.mode.upper()}]", fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel("Longitude Coordinates", fontsize=11)
        ax.set_ylabel("Latitude Coordinates", fontsize=11)
        
        save_path = f"{self.output_dir}/portfolio_spatial_accumulation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully exported contextual geographical map to: {save_path}")


    def plot_advanced_spatial_accumulation(self, damage_matrix_path=None):
        """
        Generates an executive-ready dual-panel dashboard showcasing absolute spatial
        exposure limits alongside non-linear asset accumulation curves.
        """
        # 1. Dynamically pull from self if no explicit path string was passed
        if damage_matrix_path is None:
            # Adjust this to match whatever attribute stores your damage parquet file path
            damage_matrix_path = getattr(self, 'damage_matrix_path', "./outputs/individual/damage_matrix.parquet")
        
        # 2. Now os.path.exists receives a string path, NOT the self object
        if not os.path.exists(damage_matrix_path):
            print(f"Error: Matrix file not found at {damage_matrix_path}")
            return

        # 1. Load and aggregate the streamed simulation matrix
        df = pd.read_parquet(damage_matrix_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Check spatial anchor naming
        anchor_col = [c for c in df.columns if 'anchor' in c or 'zone' in c]
        if not anchor_col:
            print("Could not find a spatial anchor column in the dataset.")
            return
        
        # Calculate a proxy for Ground-Up Loss and Insured Loss if not explicitly present
        # to maintain strict alignment with your production streaming schema
        if 'ground_up_loss' not in df.columns:
            df['ground_up_loss'] = df['wind_speed_knots'] * df['damage_ratio'] * 10_000
        if 'insured_loss' not in df.columns:
            df['insured_loss'] = df['ground_up_loss'] * 0.8  # Assume standard 20% retention/deductible
            
        summary = df.groupby(anchor_col[0]).agg({
            'ground_up_loss': 'sum',
            'insured_loss': 'sum'
        }).reset_index()
        
        # Crucial step: Sort zones descending by total exposure to reveal systemic hot-spots
        summary = summary.sort_values(by='ground_up_loss', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative metrics for risk tracking
        summary['cum_ground_up'] = summary['ground_up_loss'].cumsum()
        summary['cum_insured'] = summary['insured_loss'].cumsum()
        
        total_gu = summary['ground_up_loss'].sum()
        summary['cum_percentage'] = (summary['cum_insured'] / total_gu) * 100

        # 2. Initialize Modern Dual-Panel Layout
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Palette definition
        color_gu = '#1f4e79'       # Deep Navy
        color_ins = '#c00000'      # Crimson Red
        
        # --- Top Panel: Absolute Structural Loss Profile ---
        bar_width = 0.35
        index = np.arange(len(summary))
        
        ax1.bar(index - bar_width/2, summary['ground_up_loss'] / 1e6, bar_width, 
                label='Ground-Up Loss (Gross)', color=color_gu, alpha=0.85, edgecolor='none')
        ax1.bar(index + bar_width/2, summary['insured_loss'] / 1e6, bar_width, 
                label='Insured Loss (Net)', color=color_ins, alpha=0.9, edgecolor='none')
        
        ax1.set_ylabel('Total Losses ($ Millions)', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_title('Portfolio Spatial Risk Concentration & Financial Accumulation', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # --- Bottom Panel: Cumulative Financial Exceedance Curve ---
        ax2.plot(summary[anchor_col[0]], summary['cum_ground_up'] / 1e6, 
                color=color_gu, linewidth=2.5, linestyle='-', marker='o', label='Cumulative Gross Accumulation')
        ax2.plot(summary[anchor_col[0]], summary['cum_insured'] / 1e6, 
                color=color_ins, linewidth=2.5, linestyle='--', marker='s', label='Cumulative Net Insured Exposure')
        
        # Secondary Y-Axis for percentage curve tracking
        ax3 = ax2.twinx()
        ax3.plot(summary[anchor_col[0]], summary['cum_percentage'], color='#ffc000', linewidth=2, linestyle=':', label='% Total Portfolio Limit')
        ax3.set_ylabel('% Total Portfolio Limits Merged', color='#7f7f7f', fontsize=11, labelpad=10)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}%'))
        ax3.grid(False) # Prevent overlapping gridlines
        
        ax2.set_xlabel('Spatial Anchor Regions (Ordered by Exposure Risk)', fontsize=11, fontweight='bold', labelpad=12)
        ax2.set_ylabel('Accumulated Loss ($ Millions)', fontsize=11, fontweight='bold', labelpad=10)
        
        # Combine legends from dual twin axes smoothly
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc='lower right', frameon=True, facecolor='white')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Clean up x-axis labels layout
        plt.xticks(index, summary[anchor_col[0]], rotation=35, ha='right', fontsize=10)
        
        plt.tight_layout()
        output_png = "./data/spatial/portfolio_spatial_accumulation_upgraded.png"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"SUCCESS: Professional dashboard generated and saved to: {output_png}")
        plt.show()


    def run_all_visualizations(self):
        print(f"\n=======================================================")
        print(f"RUNNING ALL DASHBOARD GENERATIONS IN [{self.mode.upper()}] MODE")
        print(f"=======================================================")
        self.plot_exceedance_probability_curve()
        self.plot_portfolio_loss_composition()
        self.plot_model_resolution_comparison()
        self.plot_tail_risk_tvar()
        self.plot_geographic_accumulation_risk()
        self.plot_advanced_spatial_accumulation("./data/individual/portfolio_damage_matrix.parquet")
        print(f"\nPipeline successfully completed! Check folder: {self.output_dir}/")

if __name__ == "__main__":
    visualizer = CatRiskVisualizer()
    visualizer.run_all_visualizations()