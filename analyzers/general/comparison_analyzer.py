import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


class RunComparisonAnalyzer:
    """Generate physics-meaningful comparisons between modified and unmodified runs."""
    
    def __init__(self, mod_stats: Dict, unm_stats: Dict, run_name: str):
        """
        Initialize with aggregate statistics from both samples.
        
        Args:
            mod_stats: Statistics dict from modified sample
            unm_stats: Statistics dict from unmodified sample
            run_name: Name of the run (e.g., 'run_1')
        """
        self.mod_stats = mod_stats
        self.unm_stats = unm_stats
        self.run_name = run_name
    
    def plot_pt_broadening_ratio(self, output_file: Optional[str] = None) -> None:
        """
        PRIMARY OBSERVABLE: pT broadening ratio (mod σ_pT / unm σ_pT).
        
        Physics: Multiple scattering increases pT width.
        Ratio > 1.05 = significant broadening signature.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        # Core observable: σ_pT ratio
        if unm_pt['std'] > 0:
            ratio = mod_pt['std'] / unm_pt['std']
        else:
            ratio = 1.0
        
        x = np.array([0.5, 1.5])
        stds = [mod_pt['std'], unm_pt['std']]
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        bars = ax.bar(x, stds, width=0.6, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2.5)
        
        # Add numeric labels
        for bar, std in zip(bars, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{std:.4f} GeV', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        # Highlight the ratio - this is what we're measuring
        detection_level = 1.05  # 5% threshold for detection
        color = 'green' if ratio > detection_level else 'orange'
        ax.axhline(unm_pt['std'], color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (unmodified)')
        
        ax.set_ylabel(r'$\sigma_{p_T}$ (GeV)', fontsize=13, fontweight='bold')
        ax.set_title(f'pT Broadening Observable ({self.run_name})\n'
                    r'Cumulative scattering → wider $p_T$ spectrum', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)
        
        # Add ratio annotation
        ax.text(0.98, 0.95, f'Ratio σ_pT(mod) / σ_pT(unm): {ratio:.4f}\n' + 
                ('✓ Broadening detected' if ratio > detection_level else '✗ No broadening'),
                transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mean_pt_shift(self, output_file: Optional[str] = None) -> None:
        """
        SECONDARY OBSERVABLE: Mean pT shift.
        
        Physics: Cumulative effects can shift mean pT (energy loss vs scattering).
        Shows if distribution is just broader or also shifted.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        x = np.array([0.5, 1.5])
        means = [mod_pt['mean'], unm_pt['mean']]
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        bars = ax.bar(x, means, width=0.6, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2.5)
        
        # Add numeric labels
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean:.4f} GeV', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        # Calculate shift
        mean_shift = (mod_pt['mean'] - unm_pt['mean']) / unm_pt['mean'] * 100 if unm_pt['mean'] > 0 else 0
        
        ax.set_ylabel(r'Mean $p_T$ (GeV)', fontsize=13, fontweight='bold')
        ax.set_title(f'Mean pT Shift ({self.run_name})\n'
                    'Positive shift = energy gain, Negative = energy loss', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        ax.text(0.98, 0.95, f'Mean shift: {mean_shift:+.2f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_distribution_tails(self, output_file: Optional[str] = None) -> None:
        """
        TERTIARY OBSERVABLE: pT range (min to max).
        
        Physics: Cumulative scattering produces extreme particles.
        Shows whether spectrum extends to higher pT (multiple scattering signature).
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        x = np.array([0.5, 1.5])
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        # Show min-max range
        for i, (x_pos, label, color, pt_dict) in enumerate(zip(x, labels, colors, [mod_pt, unm_pt])):
            # Draw vertical line from min to max
            ax.plot([x_pos, x_pos], [pt_dict['min'], pt_dict['max']], 
                   color=color, linewidth=8, alpha=0.6, label=label)
            # Mark min, mean, max
            ax.scatter([x_pos], [pt_dict['min']], color=color, s=150, marker='v', 
                      edgecolor='black', linewidth=2, zorder=5, label=f'{label} min')
            ax.scatter([x_pos], [pt_dict['mean']], color=color, s=200, marker='o', 
                      edgecolor='black', linewidth=2, zorder=5)
            ax.scatter([x_pos], [pt_dict['max']], color=color, s=150, marker='^', 
                      edgecolor='black', linewidth=2, zorder=5, label=f'{label} max')
        
        # Max pT ratio (signature of hard scattering)
        max_ratio = mod_pt['max'] / unm_pt['max'] if unm_pt['max'] > 0 else 1.0
        
        ax.set_ylabel(r'$p_T$ (GeV)', fontsize=13, fontweight='bold')
        ax.set_title(f'pT Distribution Extent ({self.run_name})\n'
                    r'Extended high-$p_T$ tail = multiple scattering signature', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        ax.text(0.98, 0.95, f'Max pT ratio (mod/unm): {max_ratio:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rapidity_eta_spectra(self, output_file: Optional[str] = None) -> None:
        """
        QUATERNARY: Rapidity and pseudorapidity distributions.
        
        Physics: Multiple scattering affects forward/backward asymmetry.
        Shows if cumulative effects are isotropic or directional.
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        mod_y = self.mod_stats['y']
        unm_y = self.unm_stats['y']
        mod_eta = self.mod_stats['eta']
        unm_eta = self.unm_stats['eta']
        
        x = np.array([0.5, 1.5])
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        # Rapidity width
        y_stds = [mod_y['std'], unm_y['std']]
        ax1.bar(x, y_stds, width=0.6, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=2)
        for bar, std in zip(ax1.patches, y_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax1.set_ylabel(r'Rapidity width $\sigma_y$', fontsize=12, fontweight='bold')
        ax1.set_title('Rapidity Distribution Width', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Pseudorapidity width
        eta_stds = [mod_eta['std'], unm_eta['std']]
        ax2.bar(x, eta_stds, width=0.6, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=2)
        for bar, std in zip(ax2.patches, eta_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{std:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel(r'Pseudorapidity width $\sigma_\eta$', fontsize=12, fontweight='bold')
        ax2.set_title('Pseudorapidity Distribution Width', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Rapidity/Pseudorapidity Spectra ({self.run_name})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_comparisons(self, output_dir: Path) -> List[str]:
        """Generate all physics-meaningful comparison plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = []
        
        # PRIMARY: pT broadening (the main observable)
        pt_broadening_file = output_dir / "01_pt_broadening_primary.png"
        self.plot_pt_broadening_ratio(str(pt_broadening_file))
        plots.append("01_pt_broadening_primary.png")
        
        # SECONDARY: Mean pT shift
        mean_shift_file = output_dir / "02_mean_pt_shift.png"
        self.plot_mean_pt_shift(str(mean_shift_file))
        plots.append("02_mean_pt_shift.png")
        
        # TERTIARY: pT tails/extent
        tails_file = output_dir / "03_pt_distribution_tails.png"
        self.plot_distribution_tails(str(tails_file))
        plots.append("03_pt_distribution_tails.png")
        
        # QUATERNARY: Rapidity/eta spectra
        spectra_file = output_dir / "04_rapidity_eta_spectra.png"
        self.plot_rapidity_eta_spectra(str(spectra_file))
        plots.append("04_rapidity_eta_spectra.png")
        
        return plots