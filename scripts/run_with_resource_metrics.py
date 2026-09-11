#!/usr/bin/env python3
"""Run one command and write process-level resource metrics as CSV."""

import argparse
import csv
import resource
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with args.log.open("w") as log:
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
    return_code = process.wait()
    wall_seconds = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = usage.ru_utime + usage.ru_stime

    row = {
        "return_code": return_code,
        "wall_time_seconds": wall_seconds,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "cpu_time_seconds": cpu_seconds,
        "average_cpu_percent": 100.0 * cpu_seconds / wall_seconds,
        "peak_rss_mb": usage.ru_maxrss / 1024.0,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "filesystem_input_blocks": usage.ru_inblock,
        "filesystem_output_blocks": usage.ru_oublock,
    }
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
