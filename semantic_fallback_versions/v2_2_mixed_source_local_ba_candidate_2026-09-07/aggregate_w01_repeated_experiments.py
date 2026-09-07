#!/usr/bin/env python3
import argparse, csv, pathlib, statistics, json

def nums(path, col):
    vals=[]
    with path.open() as f:
        for r in csv.DictReader(f):
            try: vals.append(float(r[col]))
            except (KeyError,TypeError,ValueError): pass
    return vals

def percentile(values, q):
    if not values: return ""
    values=sorted(values); return values[min(len(values)-1, max(0, int((len(values)-1)*q)))]

def main():
    p=argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--output",default="aggregate_metrics.csv"); a=p.parse_args()
    root=pathlib.Path(a.root); rows=[]
    for group in sorted(root.glob("w01_semantic_fallback_repeat5_*")):
        for run in sorted(group.glob("run_*")):
            stats=run/"frame_stats_semantic.csv"; times=run/"frame_times_semantic.csv"; summary=run/"run_summary.csv"
            if not stats.exists(): continue
            vals=nums(times,"track_time_sec") if times.exists() else nums(stats,"track_time_ms")
            frames=nums(stats,"index")
            states=[]
            with stats.open() as sf:
                for record in csv.DictReader(sf): states.append(record.get("tracking_state", ""))
            summary_values={}
            if summary.exists():
                with summary.open() as sf:
                    summary_values=next(csv.DictReader(sf), {})
            lifecycle_counts={}
            lifecycle=run/"semantic_lifecycle_events.csv"
            if lifecycle.exists():
                with lifecycle.open() as lf:
                    for event in csv.DictReader(lf):
                        reason=event.get("reason", "unknown")
                        lifecycle_counts[reason]=lifecycle_counts.get(reason, 0)+1
            backend=run/"backend_interval_stats.csv"
            backend_last={}
            if backend.exists():
                with backend.open() as bf:
                    for backend_last in csv.DictReader(bf): pass
            metadata={}
            metadata_path=run/"run_metadata.json"
            if metadata_path.exists():
                metadata=json.loads(metadata_path.read_text())
            rows.append({"group":group.name,"run":run.name,"processed_frames":len(frames),
                         "coverage_proxy":len(set(frames)),"ok_frames":states.count("2"),
                         "recently_lost_frames":states.count("3"),"lost_frames":states.count("4"),
                         "track_mean_ms":statistics.mean(vals)*1000 if vals else "",
                         "track_median_ms":statistics.median(vals)*1000 if vals else "",
                         "track_p95_ms":percentile(vals,.95)*1000 if vals else "",
                         "track_p99_ms":percentile(vals,.99)*1000 if vals else "",
                         "atlas_maps":summary_values.get("atlas_maps", ""),
                         "current_map_keyframes":summary_values.get("current_map_keyframes", ""),
                         "current_map_points":summary_values.get("current_map_points", ""),
                         "static_map_points":summary_values.get("static_map_points", ""),
                         "provisional_map_points":summary_values.get("provisional_map_points", ""),
                         "promoted_map_points":summary_values.get("promoted_map_points", ""),
                         "promotion_count":summary_values.get("promotion_count", ""),
                         "rejection_count":summary_values.get("rejection_count", ""),
                         "lifecycle_promotion_events":lifecycle_counts.get("promotion", 0),
                         "lifecycle_expired_events":lifecycle_counts.get("expired", 0),
                         "lifecycle_low_match_events":lifecycle_counts.get("low_match_ratio", 0),
                         "lifecycle_dynamic_events":lifecycle_counts.get("dynamic_limit", 0),
                         "loop_candidates":backend_last.get("loop_candidates", ""),
                         "loop_matches":backend_last.get("loop_matches", ""),
                         "loop_trusted_inliers":backend_last.get("loop_trusted_inliers", ""),
                         "loop_provisional_inliers":backend_last.get("loop_provisional_inliers", ""),
                         "returncode":metadata.get("returncode", "")})
    with (root/a.output).open("w",newline="") as f:
        fields=["group","run","processed_frames","coverage_proxy","ok_frames","recently_lost_frames","lost_frames","track_mean_ms","track_median_ms","track_p95_ms","track_p99_ms","atlas_maps","current_map_keyframes","current_map_points","static_map_points","provisional_map_points","promoted_map_points","promotion_count","rejection_count","lifecycle_promotion_events","lifecycle_expired_events","lifecycle_low_match_events","lifecycle_dynamic_events","loop_candidates","loop_matches","loop_trusted_inliers","loop_provisional_inliers","returncode"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__=="__main__": main()
