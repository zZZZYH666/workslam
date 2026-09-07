#!/usr/bin/env python3
"""Reproducible runner for the W01 9210-frame A/B/provisional matrix.

The script intentionally does not guess dataset paths.  Use --dry-run to
review commands, then run it on the fixed evaluation host.
"""
import argparse, datetime as dt, hashlib, json, os, pathlib, shutil, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_FILES = ["SemanticFallback.h", "Frame.cc", "Tracking.cc", "LocalMapping.cc", "LoopClosing.cc", "ORBmatcher.cc", "MapPoint.h", "MapPoint.cc", "Settings.h", "Settings.cc"]

def sha256(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--vocabulary",required=True); p.add_argument("--dataset",required=True)
    p.add_argument("--masks",required=True); p.add_argument("--timestamps",required=True); p.add_argument("--output-root",required=True)
    p.add_argument("--runs",type=int,default=5); p.add_argument("--cpu-list",default="0-7"); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(); out=pathlib.Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    exe=ROOT/"third_party/ORB_SLAM3/Examples/Stereo"; baseline=exe/"finnforest_stereo_baseline"; semantic=exe/"finnforest_stereo_semantic"
    groups={"baseline":(baseline,ROOT/"configs/finnforest_w01_stereo_rectified.yaml",False),
            "candidate":(semantic,ROOT/"configs/finnforest_w01_stereo_rectified_fallback_hard.yaml",True),
            "provisional":(semantic,ROOT/"configs/finnforest_w01_stereo_rectified_fallback.yaml",True),
            "improved":(semantic,ROOT/"configs/finnforest_w01_stereo_rectified_provisional_ape_fix.yaml",True)}
    env=os.environ.copy(); env.update({k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OPENCV_FOR_THREADS_NUM")}); env["OPENBLAS_MAIN_FREE"]="1"
    def command_output(command):
        try:
            return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
        except OSError as exc:
            return f"unavailable: {exc}"
    manifest={"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"runs":a.runs,"cpu_list":a.cpu_list,
              "host":{"lscpu":command_output(["lscpu"]),"compiler":command_output(["c++","--version"]),
                      "opencv":command_output(["pkg-config","--modversion","opencv4"]),
                      "eigen":"/usr/include/eigen3"},"groups":{}}
    for name,(binary,config,is_semantic) in groups.items():
        gd=out/f"w01_semantic_fallback_repeat5_{name}"; gd.mkdir(parents=True,exist_ok=True)
        source_hashes={f:str(sha256(ROOT/"third_party/ORB_SLAM3"/("include" if f.endswith(".h") else "src")/f)) for f in SOURCE_FILES}
        manifest["groups"][name]={"binary":str(binary),"config":str(config),"source_hashes":source_hashes}
        for i in range(1,a.runs+1):
            rd=gd/f"run_{i:02d}"; rd.mkdir(parents=True,exist_ok=True)
            shutil.copy2(config, rd/"config_used.yaml")
            (rd/"source_hashes.json").write_text(json.dumps(source_hashes,indent=2))
            cmd=["taskset","-c",a.cpu_list,str(binary),a.vocabulary,str(config),a.dataset]
            if is_semantic: cmd += [a.masks,a.timestamps,str(rd)]
            else: cmd += [a.timestamps,str(rd)]
            if a.dry_run: print(" ".join(cmd)); continue
            started=time.time();
            with (rd/"run.log").open("w") as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=False)
            (rd/"run_metadata.json").write_text(json.dumps({"command":cmd,"returncode":result.returncode,"start_epoch":started,"end_epoch":time.time(),"cpu_list":a.cpu_list,"environment":{k:env[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OPENCV_FOR_THREADS_NUM")}},indent=2))
    if not a.dry_run: (out/"manifest.json").write_text(json.dumps(manifest,indent=2))

if __name__=="__main__": main()
