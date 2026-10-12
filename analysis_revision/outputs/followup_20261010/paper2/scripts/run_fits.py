"""Run a list of fit jobs with bounded concurrency. Usage: python run_fits.py JOBSET [max_parallel]."""
import itertools
import os
import subprocess
import sys
import time

from p2common import DONORS, OUT, SEEDS

PY = sys.executable
MT = ["MT_day7", "MT_day1"]
LIO = [f"LIO_{d}" for d in DONORS]
COUPLINGS = {"cfm_independent": "cfm_independent", "cfm_ot": "cfm_ot", "cfm_sb": "cfm_sb"}


def jobs(name):
    out = []
    if name == "e1":
        for task, (tag, m), seed in itertools.product(MT + LIO, COUPLINGS.items(), SEEDS):
            out.append(dict(rep="mu_std", task=task, method=m, tag=tag, seed=seed, set=[]))
        for task, seed in itertools.product(MT + LIO, SEEDS):  # optimization check: 5x longer training
            out.append(dict(rep="mu_std", task=task, method="cfm_independent", tag="cfm_independent_40k",
                            seed=seed, set=["steps=40000"]))
    elif name == "e2":
        for rep, task, m, seed in itertools.product(["mu_raw", "mu_white", "mu_clr_std", "pca30"], MT + LIO,
                                                    ["cfm_independent", "cfm_ot"], SEEDS):
            out.append(dict(rep=rep, task=task, method=m, tag=m, seed=seed, set=[]))
    elif name == "e3b":
        from p2common import inner_tasks
        for task in inner_tasks():
            for m, h, st, lr in itertools.product(["cfm_independent", "cfm_ot", "cfm_sb"], [64, 128, 256],
                                                  [2000, 8000, 20000], ["1e-3", "3e-4"]):
                # coupling cost (CPU exact OT / GPU Sinkhorn per batch) restricts the OT and SB sub-grids; see RESULTS.md
                if m == "cfm_ot" and (st == 20000 or h == 128):
                    continue
                if m == "cfm_sb" and (st == 20000 or h != 256 or lr != "1e-3"):
                    continue
                out.append(dict(rep="mu_std", task=task, method=m, tag=f"{m}_h{h}_st{st}_lr{lr}", seed=0,
                                set=[f"hidden={h}", f"steps={st}", f"lr={lr}"]))
    elif name == "e6":
        from p2common import inner_tasks
        for task in inner_tasks():
            out.append(dict(rep="mu_std", task=task, method="cfm_independent", tag="cfm_independent", seed=0, set=[]))
    return out


def done(j):
    return (OUT / "fits" / j["rep"] / j["task"] / f"{j['tag']}_s{j['seed']}.npz").exists()


if __name__ == "__main__":
    todo = [j for j in jobs(sys.argv[1]) if not done(j)]
    # slow couplings first so the queue drains evenly
    todo.sort(key=lambda j: (j["method"] == "cfm_independent", j["task"]))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    running = []
    log = open(OUT.parent / "logs" / f"fits_{sys.argv[1]}.log", "a")
    print(f"{len(todo)} jobs", flush=True)
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < limit:
            j = todo.pop(0)
            cmd = [PY, "fit_worker.py", "--rep", j["rep"], "--task", j["task"], "--method", j["method"],
                   "--seed", str(j["seed"]), "--tag", j["tag"]]
            if j["set"]:
                cmd += ["--set", *j["set"]]
            running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                            cwd=os.path.dirname(os.path.abspath(__file__))))
        time.sleep(2)
    print("all done", flush=True)
