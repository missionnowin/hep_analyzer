#!/usr/bin/env python3
# run_all_analysis.py - COMPLETE WITH LIVE PROGRESS DISPLAY
# Unified parallel analysis pipeline for Au+Au 10 GeV
# WITH BATCH PROCESSING + LIVE MULTI-THREAD PROGRESS DISPLAY


from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional
import multiprocessing as mp

import numpy as np

try:
    from analyzers.cumulative.cumulative_detector import CumulativeEffectDetector
    from analyzers.general.comparison_analyzer import RunComparisonAnalyzer
    from analyzers.general.collision_system_detector import CollisionSystemDetector
    from utils.readers import OscarReader
    from analyzers.general.aggregate_analyzer import AggregateAnalyzer
    from utils.progress_display import ProgressDisplay
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class UnifiedAnalysisEngine:
    """
    Single entry point for complete collision analysis pipeline.
    WITH MEMORY-EFFICIENT BATCH PROCESSING AND LIVE PROGRESS DISPLAY.
    """

    def __init__(self, data_root: Path, results_root: Path, n_workers: Optional[int] = None, batch_size: int = 500):
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.n_workers = n_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.results = {}


    def _process_file_batched(
        self,
        file_path: Path,
        run_name: str,
        sample_type: str,
        batch_size: int,
        system_info: Dict = None,
        progress_callback=None,
    ) -> tuple:
        """
        Process file in batches to save memory.
        
        Args:
            file_path: Path to .f19 file
            run_name: Run identifier
            sample_type: "modified" or "unmodified"
            batch_size: Events per batch
            system_info: Pre-detected collision system info (detect if None)
            progress_callback: Progress update function
        
        Returns: (aggregator, first_event, total_events, system_info)
        """
        # Detect collision system ONLY on first call (modified sample)
        if system_info is None:
            system_info = CollisionSystemDetector.detect_from_file(
                str(file_path), n_events_sample=10
            )
            if not system_info:
                system_info = {}  # Use defaults
        
        # Create aggregator with detected parameters
        aggregator = AggregateAnalyzer(
            system_label=system_info.get('label'),
            sqrt_s_NN=system_info.get('sqrt_s_NN'),
            A1=system_info.get('A1'),
            Z1=system_info.get('Z1'),
            A2=system_info.get('A2'),
            Z2=system_info.get('Z2')
        )
        
        reader = OscarReader(str(file_path))
        first_event = None
        total_events = 0
        
        # Read file in batches
        batch = []
        for event_idx, particles in enumerate(reader.read_file()):
            if first_event is None:
                first_event = particles.copy() if particles else None
            
            batch.append(particles)
            
            if len(batch) >= batch_size:
                # Process batch
                for event_particles in batch:
                    aggregator.accumulate_event(event_particles)
                total_events += len(batch)
                batch = []
                
                # Update progress
                if progress_callback:
                    progress_callback(f"Reading {sample_type:8s}: {total_events:6d} events")
        
        # Process remaining events
        for event_particles in batch:
            aggregator.accumulate_event(event_particles)
        total_events += len(batch)
        
        return aggregator, first_event, total_events, system_info

    @staticmethod
    def _analyze_single_run(
        run_num: int,
        data_root: Path,
        results_root: Path,
        batch_size: int,
        progress_callback=None,
        lock=None,
        states=None,
        total=None
    ) -> Dict:
        """
        Analyze all events in one run pair (modified + unmodified).
        """
        run_name = f"run_{run_num}"
        mod_file = data_root / "modified" / f"{run_name}.f19"
        unm_file = data_root / "no_modified" / f"{run_name}.f19"
        
        def report_progress(action: str, progress: str = ""):
            """Send progress update via callback with run_num."""
            if progress_callback:
                progress_callback(run_num, action, progress, lock, states, total)
        
        result = {
            'run_num': run_num,
            'run_name': run_name,
            'success': False,
            'error': None,
            'modified_file': str(mod_file),
            'unmodified_file': str(unm_file),
            'files_exist': {'modified': mod_file.exists(), 'unmodified': unm_file.exists()},
            'general_stats': {},
            'cumulative_analysis': {},
            'plots_generated': [],
            'system_info': {}
        }

        try:
            report_progress("Reading modified", "Initializing...")
            
            if not mod_file.exists():
                result['error'] = f"Modified file not found: {mod_file}"
                report_progress("FAILED", "File not found")
                return result

            engine = UnifiedAnalysisEngine(data_root, results_root, batch_size=batch_size)
            
            agg_mod, first_event_mod, n_events_mod, system_info = engine._process_file_batched(
                mod_file, run_name, "modified", batch_size,
                system_info=None,
                progress_callback=lambda d: report_progress("Reading modified", d)
            )
            
            result['system_info'] = system_info
            
            if n_events_mod == 0:
                result['error'] = "No events found in modified file"
                report_progress("FAILED", "No events")
                return result
            
            report_progress("Analyzing modified", f"{n_events_mod:,d} events")
            result['general_stats']['modified'] = agg_mod.get_statistics()

            agg_unm = None
            first_event_unm = None
            if unm_file.exists():
                report_progress("Reading unmodified", "Initializing...")
                
                agg_unm, first_event_unm, n_events_unm, _ = engine._process_file_batched(
                    unm_file, run_name, "unmodified", batch_size,
                    system_info=system_info,
                    progress_callback=lambda d: report_progress("Reading unmodified", d)
                )
                
                if n_events_unm > 0:
                    result['general_stats']['unmodified'] = agg_unm.get_statistics()
                    report_progress("Analyzing unmodified", f"{n_events_unm:,d} events")

            report_progress("Plotting modified", "generating...")
            out_dir_mod = results_root / "modified" / run_name
            plots_mod = agg_mod.plot_distributions(out_dir_mod)
            result['plots_generated'].extend([f"modified/{run_name}/{p}" for p in plots_mod])
            report_progress("Plotting modified", "✓ done")

            if agg_unm:
                report_progress("Plotting unmodified", "generating...")
                out_dir_unm = results_root / "unmodified" / run_name
                plots_unm = agg_unm.plot_distributions(out_dir_unm)
                result['plots_generated'].extend([f"unmodified/{run_name}/{p}" for p in plots_unm])
                report_progress("Plotting unmodified", "✓ done")

            if agg_unm:
                report_progress("Comparing", "generating...")
                comparator = RunComparisonAnalyzer(
                    result['general_stats']['modified'],
                    result['general_stats']['unmodified'],
                    run_name
                )
                out_dir_cmp = results_root / "comparisons" / run_name
                comparison_plots = comparator.generate_all_comparisons(out_dir_cmp)
                result['plots_generated'].extend([f"comparisons/{run_name}/{p}" for p in comparison_plots])
                report_progress("Comparing", "✓ done")

            if first_event_mod:
                report_progress("Detecting cumulative", "analyzing...")
                detector = CumulativeEffectDetector(first_event_mod, first_event_unm)
                detector.detect_all_signatures()
                cum_likelihood = detector.get_cumulative_likelihood()

                result['cumulative_analysis'] = {
                    'likelihood': float(cum_likelihood),
                    'n_signatures': len(detector.signatures),
                    'signatures': [
                        {
                            'type': sig.signature_type,
                            'strength': float(sig.strength),
                            'confidence': float(sig.confidence),
                            'affected_particles': sig.affected_particles,
                            'description': sig.description
                        }
                        for sig in detector.signatures
                    ]
                }
                report_progress("Detecting cumulative", f"{len(detector.signatures)} found")

            result['success'] = True
            report_progress("✓ COMPLETED", f"{system_info.get('label', 'Unknown')}")

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            report_progress("✗ FAILED", str(e)[:50])

        return result


    def run_parallel(self, runs: List[int]) -> Dict:
        """
        Process all runs in parallel with live progress display.
        """
        from utils.progress_display import ProgressDisplay
        
        print(f"\n{'='*80}")
        print("UNIFIED COLLISION ANALYSIS PIPELINE (AUTO-DETECTED SYSTEM)")
        print(f"{'='*80}")
        print(f"Runs to process: {runs}")
        print(f"Workers: {self.n_workers}")
        print(f"Batch size: {self.batch_size} events (memory efficient)\n")
        print("Processing runs in parallel...")
        print("-" * 80)
        
        # Initialize display and get shared objects
        lock, states, total = ProgressDisplay.initialize(len(runs))
        
        start_time = time.time()
        completed_runs = []
        failed_runs = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            analyze_func = partial(
                self._analyze_single_run,
                data_root=self.data_root,
                results_root=self.results_root,
                batch_size=self.batch_size,
                progress_callback=ProgressDisplay.report,
                lock=lock,
                states=states,
                total=total
            )
            
            futures = {
                executor.submit(analyze_func, run_num): run_num 
                for run_num in runs
            }
            
            result_count = 0
            for future in as_completed(futures):
                result_count += 1
                run_num = futures[future]
                
                try:
                    result = future.result(timeout=300)
                    
                    if isinstance(result, dict):
                        run_name = result.get('run_name', f'run_{run_num}')
                        self.results[run_name] = result
                        
                        if result.get('success', False):
                            completed_runs.append(run_name)
                            ProgressDisplay.print_to_console(
                                f"  [{result_count}/{len(futures)}] ✓ {run_name}",
                                lock, states, total
                            )
                        else:
                            failed_runs.append(run_name)
                            error_msg = result.get('error', 'Unknown error')
                            ProgressDisplay.print_to_console(
                                f"  [{result_count}/{len(futures)}] ✗ {run_name}: {error_msg}",
                                lock, states, total
                            )
                    else:
                        failed_runs.append(f"run_{run_num}")
                        ProgressDisplay.print_to_console(
                            f"  [{result_count}/{len(futures)}] ✗ run_{run_num}: Invalid result",
                            lock, states, total
                        )
                        
                except Exception as e:
                    failed_runs.append(f"run_{run_num}")
                    ProgressDisplay.print_to_console(
                        f"  [{result_count}/{len(futures)}] ✗ run_{run_num}: {str(e)}",
                        lock, states, total
                    )

        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        self._print_summary(elapsed, completed_runs, failed_runs)
        return self.results

    def _print_summary(self, elapsed_time: float, completed_runs: List[str], failed_runs: List[str]) -> None:
        """Print analysis summary."""
        successful = [r for r in self.results.values() if r['success']]
        failed = [r for r in self.results.values() if not r['success']]

        print(f"✓ AGGREGATE ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Completed: {len(completed_runs)}/{len(self.results)} runs")
        print(f"Failed: {len(failed_runs)}\n")

        if successful:
            print("SUCCESSFUL RUNS:")
            print("-" * 80)
            for r in successful:
                mod_stats = r['general_stats']['modified']
                sys_info = r['system_info']
                print(f"\n  {r['run_name']:12} │ System: {sys_info.get('label', 'Unknown')}")
                print(f"  {'':12} │ Events: {mod_stats['n_events']:5d} │ Particles: {mod_stats['n_total_particles']:9d}")
                print(f"  {'':12} │ pT: {mod_stats['pt']['mean']:6.3f}±{mod_stats['pt']['std']:6.3f} GeV")
                
                if 'unmodified' in r['general_stats']:
                    unm_stats = r['general_stats']['unmodified']
                    pt_ratio = mod_stats['pt']['std'] / unm_stats['pt']['std'] if unm_stats['pt']['std'] > 0 else 1.0
                    print(f"  {'':12} │ pT_std ratio (mod/unm): {pt_ratio:.3f}")
                    print(f"  {'':12} │ Cumulative likelihood: {r['cumulative_analysis']['likelihood']:.2f}")
                    print(f"  {'':12} │ Signatures: {r['cumulative_analysis']['n_signatures']}")

        if failed:
            print("\nFAILED RUNS:")
            print("-" * 80)
            for r in failed:
                print(f"  {r['run_name']:12} │ Error: {r['error']}")

        print(f"\n{'='*80}\n")


    def save_summary_json(self, output_file: Optional[str] = None) -> Path:
        """Save complete aggregate summary as JSON."""
        if not output_file:
            output_file = str(self.results_root / "aggregate_summary.json")

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {}
        for run_name, result in self.results.items():
            summary[run_name] = {
                k: v for k, v in result.items()
                if k != 'traceback'
            }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Aggregate summary saved to {output_path}")
        return output_path


    def create_cross_run_comparison_plots(self) -> None:
        """Generate cross-run comparison plots from aggregate data."""
        import matplotlib.pyplot as plt
        
        successful = [r for r in self.results.values() if r['success']]
        if not successful:
            print("No successful runs to compare")
            return

        print("\n📊 Generating cross-run comparison plot (mod vs unm)...")
        
        # Extract data for BOTH modified and unmodified samples
        run_labels = []
        pt_means = []
        pt_stds = []
        cum_likelihoods = []
        n_sigs = []
        n_events = []
        colors = []
        
        for r in successful:
            run_name = r['run_name']
            
            # MODIFIED sample
            if 'modified' in r['general_stats']:
                mod_stats = r['general_stats']['modified']
                run_labels.append(f"{run_name}\n(mod)")
                pt_means.append(mod_stats['pt']['mean'])
                pt_stds.append(mod_stats['pt']['std'])
                cum_likelihoods.append(r['cumulative_analysis']['likelihood'])
                n_sigs.append(r['cumulative_analysis']['n_signatures'])
                n_events.append(mod_stats['n_events'])
                colors.append('steelblue')
            
            # UNMODIFIED sample
            if 'unmodified' in r['general_stats']:
                unm_stats = r['general_stats']['unmodified']
                run_labels.append(f"{run_name}\n(unm)")
                pt_means.append(unm_stats['pt']['mean'])
                pt_stds.append(unm_stats['pt']['std'])
                cum_likelihoods.append(0.0)
                n_sigs.append(0)
                n_events.append(unm_stats['n_events'])
                colors.append('darkgreen')

        if not run_labels:
            print("No run data to plot")
            return

        x = np.arange(len(run_labels))
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle('Collision Analysis: Cross-Run Comparison (Modified vs Unmodified)', 
                    fontsize=15, fontweight='bold', y=0.995)

        # 1. pT means (top-left)
        ax = fig.add_subplot(2, 2, 1)
        ax.bar(x, pt_means, yerr=pt_stds, capsize=5, color=colors, 
               alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(r'Mean $p_T$ ± $\sigma_{p_T}$ (GeV)', fontsize=11, fontweight='bold')
        ax.set_title('pT with Broadening (all events)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=0, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, edgecolor='black', label='Modified'),
                          Patch(facecolor='darkgreen', alpha=0.7, edgecolor='black', label='Unmodified')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # 2. Cumulative likelihood (top-right)
        ax = fig.add_subplot(2, 2, 2)
        lik_colors = [c if cum_likelihoods[i] > 0 else 'lightgray' 
                     for i, c in enumerate(colors)]
        ax.bar(x, cum_likelihoods, color=lik_colors, alpha=0.7, 
               edgecolor='black', linewidth=1.5)
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Detection threshold')
        ax.set_ylabel('Cumulative Effect Likelihood', fontsize=11, fontweight='bold')
        ax.set_title('Cumulative Effect Detection (modified only)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=0, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # 3. Signature count (bottom-left)
        ax = fig.add_subplot(2, 2, 3)
        sig_colors = ['steelblue' if n > 0 else 'lightgray' for n in n_sigs]
        bars = ax.bar(x, n_sigs, color=sig_colors, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        for bar, n in zip(bars, n_sigs):
            if n > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(n)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Number of Cumulative Signatures', fontsize=11, fontweight='bold')
        ax.set_title('Signature Detection Count', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, rotation=0, fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # 4. Summary table (bottom-right)
        ax = fig.add_subplot(2, 2, 4)
        ax.axis('tight')
        ax.axis('off')

        table_data = [['Run', 'Type', 'Events', r'$\sigma_{p_T}$ (GeV)', 'Cum. Likelihood', 'Signals']]
        for i, label in enumerate(run_labels):
            parts = label.split('\n')
            run_num = parts[0]
            sample_type = parts[1] if len(parts) > 1 else ''
            table_data.append([
                run_num,
                sample_type,
                f"{n_events[i]:,d}",
                f"{pt_stds[i]:.3f}",
                f"{cum_likelihoods[i]:.2f}" if cum_likelihoods[i] > 0 else "—",
                f"{int(n_sigs[i])}" if n_sigs[i] > 0 else "—"
            ])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.12, 0.12, 0.16, 0.14, 0.18, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)

        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for row in range(1, len(table_data)):
            for col in range(len(table_data[0])):
                if row % 2 == 0:
                    table[(row, col)].set_facecolor('#f0f0f0')
                else:
                    table[(row, col)].set_facecolor('white')

        plt.tight_layout()
        out_file = self.results_root / "cross_run_aggregate_comparison.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_file), dpi=300, bbox_inches='tight')
        print(f"✓ Cross-run comparison plot saved to {out_file}")
        plt.close()



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified collision analysis pipeline (auto-detects collision system, memory efficient, batched)'
    )
    parser.add_argument('--data-root', type=Path, default='data',
                       help='Root directory containing modified/ and no_modified/ subdirs')
    parser.add_argument('--results-root', type=Path, default='results',
                       help='Root directory for output results')
    parser.add_argument('--runs', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                       help='Run numbers to process')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='Batch size for memory-efficient processing (default: 500)')

    args = parser.parse_args()

    engine = UnifiedAnalysisEngine(
        data_root=args.data_root,
        results_root=args.results_root,
        n_workers=args.workers,
        batch_size=args.batch_size
    )

    results = engine.run_parallel(args.runs)
    engine.save_summary_json()
    engine.create_cross_run_comparison_plots()

    print("✓ Aggregate analysis complete. Results saved to:", args.results_root)



if __name__ == '__main__':
    main()