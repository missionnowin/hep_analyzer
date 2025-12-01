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
    from utils.readers import OscarReader
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class UnifiedAnalysisEngine:
    def __init__(self, data_root: Path, results_root: Path, n_workers: Optional[int] = None, batch_size: int = 500):
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.n_workers = n_workers or mp.cpu_count()
        self.batch_size = batch_size


    def _process_files_parallel(
        self,
        file_mod: Path,
        file_unm: Path,
        run_name: str,
        batch_size: int,
        system_info: Dict = None,
        progress_callback=None,
    ) -> Tuple:
        # Detect collision system from file if not provided
        if system_info is None:
            system_info = CollisionSystemDetector.detect_from_file(
                str(file_mod), n_events_sample=10
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
        
        cumulative_analyzer = CumulativeAnalyzer()
        
        # Stream batches from both files
        reader_mod = OscarReader(str(file_mod))
        reader_unm = OscarReader(str(file_unm))
        
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
        
        cumulative_signatures = cumulative_analyzer.get_signatures()
        
        if progress_callback:
            progress_callback(
                f"Completed: {total_events:6d} event pairs, "
                f"{len(cumulative_signatures)} signatures detected"
            )
        
        return (agg_mod, agg_unm, cumulative_analyzer, cumulative_signatures, total_events, system_info)


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
                report_progress("FAILED", "File not found")
                return result
            
            if not unm_file.exists():
                result['error'] = f"Unmodified file not found: {unm_file}"
                report_progress("FAILED", "File not found")
                return result
            
            # Process files with batch streaming
            engine = UnifiedAnalysisEngine(data_root, results_root, batch_size=batch_size)
            
            agg_mod, agg_unm, cumulative_analyzer, cumulative_sigs, n_events, system_info = \
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
            # Generate plots for modified dataset
            report_progress("Plotting modified", "Generating...")
            out_dir_mod = results_root / "modified" / run_name
            out_dir_mod.mkdir(parents=True, exist_ok=True)
            agg_mod.plot_distributions(out_dir_mod)
            stats_mod = agg_mod.get_statistics()
            
            del agg_mod
            gc.collect()
            report_progress("Plotting modified", "done")

            # Process Unmodified
            report_progress("Plotting unmodified", "Generating...")
            out_dir_unm = results_root / "unmodified" / run_name
            out_dir_unm.mkdir(parents=True, exist_ok=True)
            agg_unm.plot_distributions(out_dir_unm)
            stats_unm = agg_unm.get_statistics()

            del agg_unm
            gc.collect()
            report_progress("Plotting unmodified", "done")

            # Plot cumulative analysis results
            report_progress("Plotting cumulative", "Generating...")
            out_dir_cum = results_root / "cumulative" / run_name
            out_dir_cum.mkdir(parents=True, exist_ok=True)
            cumulative_analyzer.plot_distributions(out_dir_cum)
            report_progress("Plotting cumulative", "done")

            # Compare (Using only lightweight stats)
            report_progress("Comparing", "Generating...")
            comparator = RunComparisonAnalyzer(stats_mod, stats_unm, run_name)
            out_dir_cmp = results_root / "comparisons" / run_name
            out_dir_cmp.mkdir(parents=True, exist_ok=True)
            comparator.generate_all_comparisons(out_dir_cmp)
            
            # Free comparator
            del comparator
            gc.collect()
            report_progress("Comparing", "done")

            # Save analysis results to JSON
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
                    'cumulative': cumulative_analyzer.get_statistics()
                }, f, indent=2, default=str)
            
            report_progress("Saving", "done")

            # Clean up remaining objects
            del cumulative_sigs
            del cumulative_signatures
            del cumulative_analysis
            del cumulative_analyzer
            del stats_mod
            del stats_unm
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
        print(f"\n{'='*80}")
        print("UNIFIED COLLISION ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Runs: {runs} | Workers: {self.n_workers} | Batch size: {self.batch_size}\n")

        lock, states, total = ProgressDisplay.initialize(len(runs))
        progress_callback = ProgressDisplay.report

        successful = 0
        failed = 0
        
        # Process runs in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for run_num in runs:
                future = executor.submit(
                    self._analyze_single_run,
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
            'total_events_analyzed': total_events,
            'total_signatures_detected': total_signatures,
            'signature_types': signature_types,
            'average_signature_strength': round(avg_strength, 3),
            'average_signature_confidence': round(avg_confidence, 3),
            'run_summaries': run_summaries
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary: {summary_path}")
        return summary_path


    def create_cross_run_comparison_plots(self) -> List[Path]:
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
            ax.bar(run_numbers, signatures, alpha=0.7, color='steelblue')
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Cumulative Signatures')
            ax.set_title('Cumulative Signatures Across Runs')
            ax.grid(True, alpha=0.3)
            plot_path = cross_run_dir / "signatures_per_run.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
            print(f"Plot: {plot_path.name}")
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        return plot_paths
    

    def generate_report(self) -> Path:
        report_path = self.results_root / "ANALYSIS_REPORT.txt"
        cumulative_dir = self.results_root / "cumulative"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNIFIED COLLISION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
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


def auto_detect_runs(data_root: Path) -> List[int]:
    """
    Auto-detect available run numbers from data directory.
    Looks for run_N.f19 files in modified/ subdirectory.
    
    Args:
        data_root: Root data directory
        
    Returns:
        List of run numbers found, sorted
    """
    modified_dir = data_root / "modified"
    
    if not modified_dir.exists():
        print(f"Warning: {modified_dir} not found. Using default runs [1,2,3,4,5]")
        return [1, 2, 3, 4, 5]
    
    # Find all run_N.f19 files
    run_files = sorted(modified_dir.glob("run_*.f19"))
    
    if not run_files:
        print(f"Warning: No run_*.f19 files found in {modified_dir}. Using default runs [1,2,3,4,5]")
        return [1, 2, 3, 4, 5]
    
    # Extract run numbers
    runs = []
    for f in run_files:
        try:
            # Extract number from "run_N.f19"
            run_num = int(f.stem.split('_')[1])
            runs.append(run_num)
        except (IndexError, ValueError):
            pass
    
    if runs:
        print(f"✓ Auto-detected {len(runs)} runs: {runs}")
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
                        help='Root directory containing modified/no_modified subdirectories')
    parser.add_argument('--results-root', type=Path, default=Path('results_integrated'),
                        help='Output directory for results')
    parser.add_argument('--runs', nargs='*', type=int,
                        help='Run numbers to process (auto-detected from run_*.f19 files if not specified)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Events per batch')
    
    args = parser.parse_args()
    
    # Auto-detect runs if not explicitly specified
    if args.runs is None or len(args.runs) == 0:
        print("Auto-detecting run files from data directory...")
        args.runs = auto_detect_runs(args.data_root)
    else:
        print(f"✓ Using specified runs: {args.runs}")
    
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "modified").mkdir(exist_ok=True)
    (args.results_root / "unmodified").mkdir(exist_ok=True)
    (args.results_root / "comparisons").mkdir(exist_ok=True)
    (args.results_root / "cumulative").mkdir(exist_ok=True)
    (args.results_root / "stats").mkdir(exist_ok=True)
    
    engine = UnifiedAnalysisEngine(
        data_root=args.data_root,
        results_root=args.results_root,
        n_workers=args.workers,
        batch_size=args.batch_size
    )
    
    engine.run_parallel(args.runs)
    engine.save_summary_json()
    engine.create_cross_run_comparison_plots()
    engine.generate_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results: {args.results_root}\n")


if __name__ == "__main__":
    main()
