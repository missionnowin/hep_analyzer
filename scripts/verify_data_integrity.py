"""
Data integrity verification tool - works with batched readers.
Compares modified vs unmodified files event-by-event using memory-efficient streaming.
"""

import argparse
from typing import Dict
import numpy as np
import sys

try:
    from utils.readers.reader import ReaderBase
    from utils.readers.multi_format_detector import MultiFormatReader
except ImportError:
    print("Error: Cannot import OscarReader")
    sys.exit(1)

def compare_file_pair(mod_file: str, unm_file: str, n_events: int = 5):
    """Compare modified vs unmodified files event-by-event using batch streaming."""
    
    print(f"\n{'='*80}")
    print(f"DATA INTEGRITY CHECK (Batch Streaming)")
    print(f"{'='*80}\n")
    
    print(f"Modified file:   {mod_file}")
    print(f"Unmodified file: {unm_file}\n")
    
    # Open readers with auto-detection
    reader_mod: ReaderBase = MultiFormatReader.open(mod_file)
    reader_unm: ReaderBase = MultiFormatReader.open(unm_file)
    
    # Stream events in batches
    events_mod = []
    events_unm = []
    
    batch_size = 100
    
    # Collect first n_events from each file
    for batch in reader_mod.stream_batch(batch_size):
        for event in batch:
            if len(events_mod) < n_events:
                events_mod.append(event)
            else:
                break
        if len(events_mod) >= n_events:
            break
    
    for batch in reader_unm.stream_batch(batch_size):
        for event in batch:
            if len(events_unm) < n_events:
                events_unm.append(event)
            else:
                break
        if len(events_unm) >= n_events:
            break
    
    if not events_mod or not events_unm:
        print("ERROR: Could not read files!")
        return
    
    print(f"Modified file:   {len(events_mod)} events (from first batch)")
    print(f"Unmodified file: {len(events_unm)} events (from first batch)\n")
    
    # Compare first few events
    n_compare = min(n_events, len(events_mod), len(events_unm))
    print(f"Comparing first {n_compare} events:\n")
    
    identical_count = 0
    different_count = 0
    
    for evt_idx in range(n_compare):
        evt_mod = events_mod[evt_idx]
        evt_unm = events_unm[evt_idx]
        
        n_mod = len(evt_mod)
        n_unm = len(evt_unm)
        
        # Check if events are identical
        if n_mod == n_unm:
            identical_particles = 0
            for p_mod, p_unm in zip(evt_mod, evt_unm):
                if (p_mod.px == p_unm.px and 
                    p_mod.py == p_unm.py and 
                    p_mod.pz == p_unm.pz and
                    p_mod.particle_id == p_unm.particle_id):
                    identical_particles += 1
            
            if identical_particles == n_mod:
                print(f"Event {evt_idx}: ✗ IDENTICAL ({n_mod} particles)")
                identical_count += 1
            else:
                print(f"Event {evt_idx}: ✓ Different ({identical_particles}/{n_mod} particles match)")
                different_count += 1
        else:
            print(f"Event {evt_idx}: ✓ Different multiplicity (mod: {n_mod}, unm: {n_unm})")
            different_count += 1
        
        # Show sample particles from first event
        if evt_idx == 0:
            print(f"\n  First 3 particles (Modified):")
            for i, p in enumerate(evt_mod[:3]):
                print(f"    {i}: px={p.px:7.3f} py={p.py:7.3f} pz={p.pz:7.3f} pdgid={p.particle_id}")
            
            print(f"\n  First 3 particles (Unmodified):")
            for i, p in enumerate(evt_unm[:3]):
                print(f"    {i}: px={p.px:7.3f} py={p.py:7.3f} pz={p.pz:7.3f} pdgid={p.particle_id}")
            print()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (first batch):")
    print(f"{'='*80}")
    print(f"Identical events:  {identical_count}")
    print(f"Different events:  {different_count}")
    
    if identical_count == n_compare:
        print("\n❌ CRITICAL: Modified and unmodified files are IDENTICAL!")
        print("   Check your generator/data pipeline!")
    elif different_count > 0:
        print("\n✓ Files appear to be different (as expected)")
    
    # Statistical comparison (streaming all events)
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON (streaming all events):")
    print(f"{'='*80}\n")
    
    def calc_pt_stats_streaming(reader, max_events: int = 10000) -> Dict:
        """Calculate pT statistics by streaming through file."""
        pt_all = []
        total_particles = 0
        total_events = 0
        
        for batch in reader.stream_batch(batch_size=100):
            for event in batch:
                total_events += 1
                for particle in event:
                    pt = (particle.px**2 + particle.py**2)**0.5
                    pt_all.append(pt)
                    total_particles += 1
                
                if total_events >= max_events:
                    break
            if total_events >= max_events:
                break
        
        if not pt_all:
            return {'mean': 0, 'std': 0, 'n_particles': 0, 'n_events': 0}
        
        pt_array = np.array(pt_all)
        return {
            'mean': np.mean(pt_array),
            'std': np.std(pt_array),
            'n_particles': total_particles,
            'n_events': total_events
        }
    
    # Re-open readers for full streaming
    reader_mod = MultiFormatReader.open(mod_file)
    reader_unm = MultiFormatReader.open(unm_file)
    
    stats_mod = calc_pt_stats_streaming(reader_mod, max_events=10000)
    stats_unm = calc_pt_stats_streaming(reader_unm, max_events=10000)
    
    print(f"Modified (first {stats_mod['n_events']} events):")
    print(f"  Total particles: {stats_mod['n_particles']:,d}")
    print(f"  pT mean:  {stats_mod['mean']:.4f} GeV")
    print(f"  pT std:   {stats_mod['std']:.4f} GeV")
    
    print(f"\nUnmodified (first {stats_unm['n_events']} events):")
    print(f"  Total particles: {stats_unm['n_particles']:,d}")
    print(f"  pT mean:  {stats_unm['mean']:.4f} GeV")
    print(f"  pT std:   {stats_unm['std']:.4f} GeV")
    
    if stats_unm['mean'] > 0:
        delta_mean_pct = abs(stats_mod['mean'] - stats_unm['mean']) / stats_unm['mean'] * 100
        delta_std_pct = abs(stats_mod['std'] - stats_unm['std']) / stats_unm['std'] * 100
        
        print(f"\nDifference:")
        print(f"  ΔpT_mean: {abs(stats_mod['mean'] - stats_unm['mean']):.6f} GeV ({delta_mean_pct:.2f}%)")
        print(f"  ΔpT_std:  {abs(stats_mod['std'] - stats_unm['std']):.6f} GeV ({delta_std_pct:.2f}%)")
        
        if delta_mean_pct < 0.1:
            print("\n  ⚠️  Statistics are IDENTICAL (< 0.1% difference)")
            print("      This suggests files are the same or generator has no effect")
        else:
            print(f"\n  ✓ Significant difference detected ({delta_mean_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Verify data integrity with batch streaming')
    parser.add_argument('--mod', '-m', required=True, help='Modified file path')
    parser.add_argument('--unm', '-u', required=True, help='Unmodified file path')
    parser.add_argument('--events', '-e', type=int, default=5,
                       help='Number of events to compare (default: 5)')
    parser.add_argument('--max-stream', type=int, default=10000,
                       help='Max events to stream for statistics (default: 10000)')
    
    args = parser.parse_args()
    
    compare_file_pair(args.mod, args.unm, args.events)


if __name__ == '__main__':
    main()