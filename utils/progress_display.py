"""Progress display for parallel analysis pipeline."""

class ProgressDisplay:
    """
    Static progress display - no instance state, fully picklable.
    Safe to use with multiprocessing.
    """
    
    @staticmethod
    def report(run_name: str, stage: str, details: str = "") -> None:
        """
        Report progress from worker process.
        
        Args:
            run_name: Name of the run (e.g., 'run_1')
            stage: Current stage (e.g., 'Reading modified', 'Plotting')
            details: Additional details (e.g., '5000 events', '✓ Done')
        """
        if details:
            print(f"    [{run_name}] {stage:25s} | {details}", flush=True)
        else:
            print(f"    [{run_name}] {stage}", flush=True)
