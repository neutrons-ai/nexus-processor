#!/usr/bin/env python3
"""
Mantid Script: Count Events by Time Slice

This Mantid script filters neutron detector events by time intervals and returns
the count of events in each time slice. It reads directly from NeXus HDF5 files.

The correlation works as follows:
1. Each neutron pulse has a wall-clock time recorded in the proton_charge log
2. Each event has a pulse time indicating which pulse it belongs to
3. The event's absolute time = pulse_time + time_offset

This is the Mantid equivalent of spark_events_by_time_parquet.py and provides
the same functionality using Mantid's built-in event filtering capabilities.

Usage:
    python mantid_events_by_time.py \
        --file data/REF_L_218386.nxs.h5 \
        --interval 60

    python mantid_events_by_time.py \
        --file data/REF_L_218386.nxs.h5 \
        --start 30 --end 120

Requirements:
    - Mantid framework installed (conda install -c mantid mantid)
    - NeXus file with event data

Arguments:
    --file          Path to NeXus HDF5 file
    --interval      Time interval in seconds for grouping events
    --start         Start time in seconds (requires --end)
    --end           End time in seconds (requires --start)
    --bank          Filter to specific detector bank
    --by-bank       Show counts per detector bank
    --output        Output CSV file for results
"""

import sys

import argparse
import numpy as np

import mantid.kernel
import mantid.simpleapi as api

# Parse command-line arguments for run number and time interval
parser = argparse.ArgumentParser(description="Filter Mantid events by time intervals")
parser.add_argument("--run-number", "-r", required=True, type=int,
                    help="Run number (integer) used to build the NeXus filename <instrument>_<run_number>.nxs.h5")
parser.add_argument("--time-interval", "-t", required=True, type=float,
                    help="Time interval in seconds for grouping events (e.g. 60)")
parser.add_argument("--instrument", "-i", default="REF_L", type=str,
                    help="Instrument name (default: REF_L)")
args = parser.parse_args()

# Expose variables expected by the rest of the script
run_number = args.run_number
time_interval = args.time_interval
instrument = args.instrument

print("Slicing data")
meas_ws = api.LoadEventNexus("%s_%s" % (instrument, run_number))

splitws, infows = api.GenerateEventsFilter(InputWorkspace=meas_ws, TimeInterval=time_interval)

api.FilterEvents(
    InputWorkspace=meas_ws,
    SplitterWorkspace=splitws,
    InformationWorkspace=infows,
    OutputWorkspaceBaseName="time_ws",
    GroupWorkspaces=True,
    FilterByPulseTime=True,
    OutputWorkspaceIndexedFrom1=True,
    CorrectionToSample="None",
    SpectrumWithoutDetector="Skip",
    SplitSampleLogs=False,
    OutputTOFCorrectionWorkspace="mock",
)
wsgroup = api.mtd["time_ws"]
wsnames = wsgroup.getNames()

for name in wsnames:
    tmpws = api.mtd[name]
    print("workspace %s has %d events" % (name, tmpws.getNumberEvents()))