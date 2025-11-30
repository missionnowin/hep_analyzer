#!/usr/bin/env python3
import sys
from datetime import datetime
from multiprocessing import Manager


class ProgressDisplay:
    """Live progress display - workers receive lock and dict as args."""
    
    @staticmethod
    def initialize(num_runs: int):
        """Create and return Manager lock and dict."""
        manager = Manager()
        lock = manager.Lock()
        states = manager.dict()
        
        # Initialize state but DON'T print lines yet
        for i in range(1, num_runs + 1):
            states[i] = {
                'action': 'Waiting',
                'progress': '',
                'timestamp': ''
            }
        
        # Print initial lines for each run
        for i in range(1, num_runs + 1):
            print(f"[--:--:--] [run_{i}] Waiting")
        
        return lock, states, num_runs
    
    @staticmethod
    def report(run_num: int, action: str, progress: str, lock, states, total):
        """Update display - receives lock and states as args."""
        from datetime import datetime
        
        if total == 0 or lock is None:
            return
        
        try:
            with lock:
                states[run_num] = {
                    'action': action,
                    'progress': progress,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                }
                
                # Cursor is at end after all lines printed
                # Move UP to line for run_num
                lines_up = total - run_num + 1
                sys.stdout.write(f"\033[{lines_up}A")
                sys.stdout.write("\033[K")
                
                state = states[run_num]
                if state['progress']:
                    line = f"[{state['timestamp']}] [run_{run_num}] {state['action']:20s} | {state['progress']}"
                else:
                    line = f"[{state['timestamp']}] [run_{run_num}] {state['action']:20s}"
                
                sys.stdout.write(line + "\n")
                sys.stdout.write(f"\033[{lines_up}B")
                sys.stdout.flush()
        except:
            pass
    
    @staticmethod
    def print_to_console(message: str, lock, states, total):
        """Print message below display."""
        if total == 0 or lock is None:
            print(message)
            return
        
        try:
            with lock:
                sys.stdout.write(f"\033[{total}B")
                sys.stdout.write(message + "\n")
                sys.stdout.flush()
        except:
            print(message)
