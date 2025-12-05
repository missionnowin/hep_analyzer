#!/usr/bin/env python3

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
import argparse
import gc


try:
    from analyzers.general.comparison_analyzer import RunComparisonAnalyzer
    from utils.collision_system_detector import CollisionSystemDetector
    from analyzers.general.aggregate_analyzer import AggregateAnalyzer
    from utils.progress_display import ProgressDisplay
    from analyzers.cumulative.cumulative_analyzer import CumulativeAnalyzer
    from readers.multi_format_detector import MultiFormatReader
    from readers.reader import ReaderBase
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class UnifiedAnalysisEngine:
    """
    Main analysis engine for collision data.
    
    Features:
    - Dual mode: comparison (modified/unmodified) or single (single files)
    - Batch streaming to minimize memory footprint
    - Parallel processing with ProcessPoolExecutor
    - Automatic cleanup after each phase
    """
    
    def __init__(self, data_root: Path, results_root: Path, run_mode: str = "comparison", 
                 file_format: str = None, n_workers: Optional[int] = None, batch_size: int = 500):
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.run_mode = run_mode  # 'comparison' or 'single'
        self.n_workers = n_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.file_format = file_format

    def _process_single_file(
        self,
        file_path: Path,
        run_name: str,
        batch_size: int,
        system_info: Dict = None,
        progress_callback=None,
    ) -> Tuple[AggregateAnalyzer, int, Dict]:
        """
        Process a single file for analysis.
        
        Returns:
            Tuple of (agg, cumulative_signatures, total_events, system_info)
        """
        # Detect collision system from file if not provided
        if system_info is None:
            system_info = CollisionSystemDetector.detect_from_file(
                str(file_path), file_format=self.file_format
            )
            if not system_info:
                system_info = {}
        
        # Initialize aggregator
        agg = AggregateAnalyzer(
            system_label=system_info.get('label'),
            sqrt_s_NN=system_info.get('sqrt_s_NN'),
            A1=system_info.get('A1'),
            Z1=system_info.get('Z1'),
            A2=system_info.get('A2'),
            Z2=system_info.get('Z2')
        )
        
        # Stream batches from file
        reader: ReaderBase = MultiFormatReader.open(file_path, self.file_format) 
        
        total_events = 0
        
        # Process batches: loads exactly batch_size events into memory
        for batch in reader.stream_batch(batch_size):
            # Accumulate statistics
            for event_particles in batch:
                agg.accumulate_event(event_particles)
            
            total_events += len(batch)
            
            # Clear batch from memory immediately
            batch.clear()
            gc.collect()
            
            if progress_callback:
                progress_callback(f"Processed {total_events:6d} events")
        
        return (agg, total_events, system_info)

    def _process_files_parallel(
        self,
        file_mod: Path,
        file_unm: Path,
        run_name: str,
        batch_size: int,
        system_info: Dict = None,
        progress_callback=None,
    ) -> Tuple[AggregateAnalyzer, AggregateAnalyzer, CumulativeAnalyzer, int, Dict]:
        """
        Process modified and unmodified files in parallel batches (comparison mode).
        
        Returns:
            Tuple of (agg_mod, agg_unm, cumulative_analyzer, cumulative_signatures, total_events, system_info)
        """
        # Detect collision system from file if not provided
        if system_info is None:
            system_info = CollisionSystemDetector.detect_from_file(
                str(file_mod), file_format=self.file_format
            )
            if not system_info:
                system_info = {}
        
        # Initialize aggregators for both datasets
        agg_mod = AggregateAnalyzer(
            system_label=system_info.get('label'),
            sqrt_s_NN=system_info.get('sqrt_s_NN'),
            A1=system_info.get('A1'),
            Z1=system_info.get('Z1'),
            A2=system_info.get('A2'),
            Z2=system_info.get('Z2')
        )
        
        agg_unm = AggregateAnalyzer(
            system_label=system_info.get('label'),
            sqrt_s_NN=system_info.get('sqrt_s_NN'),
            A1=system_info.get('A1'),
            Z1=system_info.get('Z1'),
            A2=system_info.get('A2'),
            Z2=system_info.get('Z2')
        )
        
        # TUNED FOR 1% FLUCTON FRACTION
        cumulative_analyzer = CumulativeAnalyzer(
            threshold_strength=0.01,       # 1% excess = 1% flucton fraction
            threshold_confidence=0.05,     # Moderate confidence for subtle effects
            threshold_absolute_excess=0
        )
        
        # Stream batches from both files
        reader_mod: ReaderBase = MultiFormatReader.open(str(file_mod), self.file_format)
        reader_unm: ReaderBase = MultiFormatReader.open(str(file_unm), self.file_format)
        
        total_events = 0
        
        # Process batches: loads exactly batch_size events into memory
        for batch_mod, batch_unm in zip(
            reader_mod.stream_batch(batch_size),
            reader_unm.stream_batch(batch_size)
        ):
            # Accumulate statistics for both datasets
            for event_particles in batch_mod:
                agg_mod.accumulate_event(event_particles)
            
            for event_particles in batch_unm:
                agg_unm.accumulate_event(event_particles)
            
            # Analyze cumulative effects
            cumulative_analyzer.process_batch(batch_mod, batch_unm)
            
            total_events += len(batch_mod)
            
            # Clear batch from memory immediately
            batch_mod.clear()
            batch_unm.clear()
            gc.collect()
            
            if progress_callback:
                progress_callback(f"Processed {total_events:6d} event pairs")
        
        return (agg_mod, agg_unm, cumulative_analyzer, total_events, system_info)

    @staticmethod
    def _analyze_single_run_comparison(
        run_num: int,
        data_root: Path,
        results_root: Path,
        batch_size: int,
        progress_callback=None,
        lock=None,
        states=None,
        total=None
    ) -> Dict:
        """Analyze a single run in comparison mode (modified vs unmodified)."""
        run_name = f"run_{run_num}"
        mod_file = data_root / "modified" / f"{run_name}.f19"
        unm_file = data_root / "no_modified" / f"{run_name}.f19"

        def report_progress(action: str, progress: str = ""):
            if progress_callback:
                progress_callback(run_num, action, progress, lock, states, total)

        result = {
            'run_num': run_num,
            'run_name': run_name,
            'success': False,
            'error': None,
        }

        try:
            report_progress("Processing", "Reading both files...")
            
            if not mod_file.exists():
                result['error'] = f"Modified file not found: {mod_file}"
                report_progress("FAILED", f"File not found {mod_file}")
                return result
            
            if not unm_file.exists():
                result['error'] = f"Unmodified file not found: {unm_file}"
                report_progress("FAILED", f"File not found: {unm_file}")
                return result
            
            # Process files with batch streaming
            engine: UnifiedAnalysisEngine = UnifiedAnalysisEngine(data_root, results_root, run_mode="comparison", batch_size=batch_size)
            
            agg_mod, agg_unm, cumulative_analyzer, n_events, system_info = \
                engine._process_files_parallel(
                    mod_file, unm_file, run_name, batch_size,
                    system_info=None,
                    progress_callback=lambda d: report_progress("Processing", d)
                )
            
            if n_events == 0:
                result['error'] = "No events found"
                report_progress("FAILED", "No events")
                return result
            
            report_progress("Analyzing", f"{n_events:,d} event pairs")
            
            # 1. Process Modified (Plot, Stats, then DELETE)
            report_progress("Plotting modified", "Generating...")
            out_dir_mod = results_root / "modified" / run_name
            out_dir_mod.mkdir(parents=True, exist_ok=True)
            agg_mod.plot_distributions(out_dir_mod)
            stats_mod = agg_mod.get_statistics()
            del agg_mod
            gc.collect()
            report_progress("Plotting modified", "done")

            # 2. Process Unmodified (Plot, Stats, then DELETE)
            report_progress("Plotting unmodified", "Generating...")
            out_dir_unm = results_root / "unmodified" / run_name
            out_dir_unm.mkdir(parents=True, exist_ok=True)
            agg_unm.plot_distributions(out_dir_unm)
            stats_unm = agg_unm.get_statistics()
            del agg_unm
            gc.collect()
            report_progress("Plotting unmodified", "done")

            # 3. Plot cumulative analysis
            report_progress("Plotting cumulative", "Generating...")
            out_dir_cum = results_root / "cumulative" / run_name
            out_dir_cum.mkdir(parents=True, exist_ok=True)
            cumulative_analyzer.plot_distributions(out_dir_cum)
            stats_cumulative, cumulative_sigs = cumulative_analyzer.get_statistics()
            report_progress("Plotting cumulative", "done")
            del cumulative_analyzer
            gc.collect()

            # 4. Compare
            report_progress("Comparing", "Generating...")
            comparator = RunComparisonAnalyzer(stats_mod, stats_unm, run_name)
            out_dir_cmp = results_root / "comparisons" / run_name
            out_dir_cmp.mkdir(parents=True, exist_ok=True)
            comparator.generate_all_comparisons(out_dir_cmp)
            del comparator
            gc.collect()
            report_progress("Comparing", "done")

            # Save results to JSON
            report_progress("Saving", "Writing cumulative analysis...")
            
            cumulative_signatures = [
                {
                    'type': sig.signature_type,
                    'strength': float(sig.strength),
                    'confidence': float(sig.confidence),
                    'affected_particles': sig.affected_particles,
                    'description': sig.description
                }
                for sig in cumulative_sigs
            ]
            
            cumulative_analysis = {
                'total_events': int(n_events),
                'signatures_detected': len(cumulative_sigs),
                'signature_types': list(set(s.signature_type for s in cumulative_sigs))
            }
            
            cum_json_path = results_root / "cumulative" / f"{run_name}_cumulative.json"
            cum_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cum_json_path, 'w') as f:
                json.dump({
                    'cumulative_analysis': cumulative_analysis,
                    'signatures': cumulative_signatures
                }, f, indent=2)
            
            # Save statistics
            stats_json_path = results_root / "stats" / f"{run_name}_stats.json"
            stats_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_json_path, 'w') as f:
                json.dump({
                    'run_name': run_name,
                    'modified': stats_mod,
                    'unmodified': stats_unm,
                    'cumulative': stats_cumulative
                }, f, indent=2, default=str)
            
            report_progress("Saving", "done")
            del cumulative_sigs, cumulative_signatures, cumulative_analysis, stats_mod, stats_unm
            gc.collect()

            result['success'] = True
            result['system_info'] = system_info
            report_progress("COMPLETED", f"{system_info.get('label', 'Unknown')}")

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            report_progress("FAILED", str(e)[:50])

        return result

    @staticmethod
    def _analyze_single_run_single(
        run_num: int,
        data_root: Path,
        results_root: Path,
        batch_size: int,
        progress_callback=None,
        lock=None,
        states=None,
        total=None
    ) -> Dict:
        """Analyze a single run in single mode (single file distributions only)."""
        run_name = f"run_{run_num}"
        file_path = data_root / "no_modified" / f"{run_name}.f19"

        def report_progress(action: str, progress: str = ""):
            if progress_callback:
                progress_callback(run_num, action, progress, lock, states, total)

        result = {
            'run_num': run_num,
            'run_name': run_name,
            'success': False,
            'error': None,
        }

        try:
            report_progress("Processing", "Reading file...")
            
            if not file_path.exists():
                result['error'] = f"File not found: {file_path}"
                report_progress("FAILED", f"File not found {file_path}")
                return result
            
            # Process file
            engine = UnifiedAnalysisEngine(data_root, results_root, run_mode="single", batch_size=batch_size)
            
            agg, n_events, system_info = \
                engine._process_single_file(
                    file_path, run_name, batch_size,
                    system_info=None,
                    progress_callback=lambda d: report_progress("Processing", d)
                )
            
            if n_events == 0:
                result['error'] = "No events found"
                report_progress("FAILED", "No events")
                return result
            
            report_progress("Analyzing", f"{n_events:,d} events")
            
            # Plot distributions
            report_progress("Plotting", "Generating...")
            out_dir = results_root / "distributions" / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.plot_distributions(out_dir)
            stats = agg.get_statistics()
            del agg
            gc.collect()
            report_progress("Plotting", "done")
            
            # Save statistics
            stats_json_path = results_root / "stats" / f"{run_name}_stats.json"
            stats_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_json_path, 'w') as f:
                json.dump({
                    'run_name': run_name,
                    'total_events': int(n_events),
                    'statistics': stats,
                }, f, indent=2, default=str)
            
            report_progress("Saving", "done")
            del stats
            gc.collect()

            result['success'] = True
            result['system_info'] = system_info
            report_progress("COMPLETED", f"{system_info.get('label', 'Unknown')}")

        except Exception as e:
            result['error'] = str(e)
            import traceback
            result['traceback'] = traceback.format_exc()
            report_progress("FAILED", str(e)[:50])

        return result

    def run_parallel(self, runs: List[int]) -> None:
        """Execute analysis on multiple runs in parallel."""
        print(f"\n{'='*80}")
        print("UNIFIED COLLISION ANALYSIS PIPELINE")
        print(f"Mode: {self.run_mode.upper()}")
        print(f"{'='*80}")
        print(f"Runs: {runs} | Workers: {self.n_workers} | Batch size: {self.batch_size}\n")

        lock, states, total = ProgressDisplay.initialize(len(runs))
        progress_callback = ProgressDisplay.report

        successful = 0
        failed = 0
        
        # Select appropriate analysis function based on run mode
        analyze_func = self._analyze_single_run_single if self.run_mode == "single" else self._analyze_single_run_comparison
        
        # Process runs in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for run_num in runs:
                future = executor.submit(
                    analyze_func,
                    run_num,
                    self.data_root,
                    self.results_root,
                    self.batch_size,
                    progress_callback=progress_callback,
                    lock=lock,
                    states=states,
                    total=total
                )
                futures[future] = run_num

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=7200)
                    if result.get('success'):
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                
                del result
                gc.collect()

        print(f"\nSuccessful: {successful}, Failed: {failed}")

    def save_summary_json(self) -> Path:
        """Generate cross-run summary statistics."""
        summary_path = self.results_root / "analysis_summary.json"
        cumulative_dir = self.results_root / "cumulative"
        
        total_events = 0
        total_signatures = 0
        signature_types = {}
        avg_strength = 0
        avg_confidence = 0
        count = 0
        run_summaries = []
        
        # Read all cumulative results from JSON
        if cumulative_dir.exists():
            for cum_file in sorted(cumulative_dir.glob("*_cumulative.json")):
                try:
                    with open(cum_file) as f:
                        cum_data = json.load(f)
                        cum_analysis = cum_data.get('cumulative_analysis', {})
                        total_events += cum_analysis.get('total_events', 0)
                        total_signatures += cum_analysis.get('signatures_detected', 0)
                        
                        for sig_type in cum_analysis.get('signature_types', []):
                            signature_types[sig_type] = signature_types.get(sig_type, 0) + 1
                        
                        for sig in cum_data.get('signatures', []):
                            avg_strength += sig.get('strength', 0)
                            avg_confidence += sig.get('confidence', 0)
                            count += 1
                        
                        run_summaries.append({
                            'run': cum_file.stem.replace('_cumulative', ''),
                            'signatures': cum_analysis.get('signatures_detected', 0)
                        })
                except Exception as e:
                    print(f"Error reading {cum_file}: {e}")
        
        if count > 0:
            avg_strength /= count
            avg_confidence /= count
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.run_mode,
            'total_events_analyzed': total_events,
            'total_signatures_detected': total_signatures,
            'signature_types': signature_types,
            'average_signature_strength': round(avg_strength, 3),
            'average_signature_confidence': round(avg_confidence, 3),
            'run_summaries': run_summaries,
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary: {summary_path}")
        return summary_path

    def create_cross_run_comparison_plots(self) -> List[Path]:
        """Generate cross-run comparison plots."""
        plot_paths = []
        cumulative_dir = self.results_root / "cumulative"
        if not cumulative_dir.exists():
            return plot_paths
        
        run_numbers = []
        signatures = []
        
        # Collect data from JSON files
        for cum_file in sorted(cumulative_dir.glob("*_cumulative.json")):
            try:
                with open(cum_file) as f:
                    cum_data = json.load(f)
                    run_num = int(cum_file.stem.replace('_cumulative', '').replace('run_', ''))
                    run_numbers.append(run_num)
                    signatures.append(cum_data['cumulative_analysis'].get('signatures_detected', 0))
            except:
                pass
        
        if len(run_numbers) < 2:
            return plot_paths
        
        try:
            import matplotlib.pyplot as plt
            cross_run_dir = self.results_root / "cross_run_analysis"
            cross_run_dir.mkdir(parents=True, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(run_numbers, signatures, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Run Number', fontsize=12)
            ax.set_ylabel('Cumulative Signatures', fontsize=12)
            title = 'Cumulative Signatures Across Runs'
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plot_path = cross_run_dir / "signatures_per_run.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            plot_paths.append(plot_path)
            print(f"Cross-run plot: {plot_path.name}")
        except Exception as e:
            print(f"Error creating cross-run plots: {e}")
        
        return plot_paths

    def generate_report(self) -> Path:
        """Generate comprehensive analysis report."""
        report_path = self.results_root / "ANALYSIS_REPORT.txt"
        cumulative_dir = self.results_root / "cumulative"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"UNIFIED COLLISION ANALYSIS REPORT ({self.run_mode.upper()} MODE)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("DETECTION THRESHOLDS:\n")
            f.write("  threshold_strength = 0.01 (1%)\n")
            f.write("  threshold_confidence = 0.05\n")
            f.write("  threshold_absolute_excess = 0 (disabled)\n\n")
            f.write("MEMORY OPTIMIZATIONS:\n")
            f.write("  - Matplotlib backend: Agg (non-interactive)\n")
            f.write("  - Thread-safe figure cleanup (plt.close('all'))\n")
            f.write("  - Aggressive garbage collection enabled\n")
            f.write("  - Batch streaming for large files\n\n")
            
            if cumulative_dir.exists():
                f.write("PER-RUN ANALYSIS\n")
                f.write("-"*80 + "\n")
                
                for cum_file in sorted(cumulative_dir.glob("*_cumulative.json")):
                    try:
                        with open(cum_file) as cf:
                            cum_data = json.load(cf)
                            run_name = cum_file.stem.replace('_cumulative', '')
                            cum_analysis = cum_data['cumulative_analysis']
                            
                            f.write(f"\n{run_name}:\n")
                            f.write(f"  Events: {cum_analysis.get('total_events', 0):,}\n")
                            f.write(f"  Signatures: {cum_analysis.get('signatures_detected', 0)}\n")
                            f.write(f"  Types: {cum_analysis.get('signature_types', [])}\n")
                    except:
                        pass
        
        print(f"Report: {report_path}")
        return report_path


def auto_detect_runs(data_root: Path, run_mode: str = "comparison") -> List[int]:
    """
    Auto-detect available run numbers from data directory.
    
    Args:
        data_root: Root data directory
        run_mode: 'comparison' (modified/unmodified) or 'single' (single files only)
        
    Returns:
        List of run numbers found, sorted
    """
    if run_mode == "single":
        # Look for run_N.f19 files directly in data_root
        run_files = sorted(data_root.glob("run_*.f19"))
        search_path = data_root
    else:
        # Look in modified/ subdirectory for comparison mode
        modified_dir = data_root / "modified"
        run_files = sorted(modified_dir.glob("run_*.f19")) if modified_dir.exists() else []
        search_path = modified_dir
    
    if not run_files:
        print(f"Warning: No run_*.f19 files found in {search_path}. Using default runs [1,2,3,4,5]")
        return [1, 2, 3, 4, 5]
    
    # Extract run numbers
    runs = []
    for f in run_files:
        try:
            run_num = int(f.stem.split('_')[1])
            runs.append(run_num)
        except (IndexError, ValueError):
            pass
    
    if runs:
        print(f"✓ Auto-detected {len(runs)} runs in {run_mode} mode: {runs}")
        return sorted(runs)
    else:
        print(f"Warning: Could not parse run numbers. Using default runs [1,2,3,4,5]")
        return [1, 2, 3, 4, 5]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified collision analysis pipeline'
    )
    
    parser.add_argument('--data-root', type=Path, default=Path('data'),
                        help='Root directory containing data files')
    parser.add_argument('--results-root', type=Path, default=Path('results'),
                        help='Output directory for results')
    parser.add_argument('--mode', type=str, choices=['comparison', 'single'], default='comparison',
                        help='Analysis mode: comparison (modified/unmodified) or single (single files)')
    parser.add_argument('--format', type=str, choices=['oscar', 'hepmc', 'phqmd', 'auto'], default='auto',
                        help='File format: oscar (.f19), hepmc (.hepmc), phqmd (.phsd.dat), or auto-detect')
    parser.add_argument('--runs', nargs='*', type=int,
                        help='Run numbers to process (auto-detected if not specified)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Events per batch')
    
    args = parser.parse_args()
    
    # Auto-detect runs if not explicitly specified
    if args.runs is None or len(args.runs) == 0:
        print(f"Auto-detecting run files from data directory ({args.mode} mode)...")
        args.runs = auto_detect_runs(args.data_root, run_mode=args.mode)
    else:
        print(f"✓ Using specified runs: {args.runs}")
    
    # Create output directories
    args.results_root.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "comparison":
        (args.results_root / "modified").mkdir(exist_ok=True)
        (args.results_root / "unmodified").mkdir(exist_ok=True)
        (args.results_root / "comparisons").mkdir(exist_ok=True)
    else:
        (args.results_root / "distributions").mkdir(exist_ok=True)
    
    (args.results_root / "cumulative").mkdir(exist_ok=True)
    (args.results_root / "stats").mkdir(exist_ok=True)
    
    # Run analysis
    engine = UnifiedAnalysisEngine(
        data_root=args.data_root,
        results_root=args.results_root,
        run_mode=args.mode,
        file_format=args.format,
        n_workers=args.workers,
        batch_size=args.batch_size
    )
    
    engine.run_parallel(args.runs)
    engine.save_summary_json()
    engine.create_cross_run_comparison_plots()
    engine.generate_report()
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE ({args.mode.upper()} Mode)")
    print("="*80)
    print(f"Results: {args.results_root}\n")


if __name__ == "__main__":
    main()

