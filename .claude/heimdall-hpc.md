```
  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
       ~~~      .-'''-.      ~~~       .-'''-.      ~~~       .-'''-.
  _.-'''-._.-'''       '''-._.-'''-._.'       '''-._.-'''-._.'       '''-.


                      H   E   I   M   D   A   L   L


  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
 ~~~~   ~~   ~~~~   ~~~   ~~   ~~~~   ~~   ~~~~   ~~~  ~~ ~~~~   ~~   ~~~~
```

# NYU TORCH HPC
Distilled from the [NYU RTS docs](https://services.rt.nyu.edu/docs/hpc/getting_started/intro/) (source repo
[NYU-RTS/rts-docs](https://github.com/NYU-RTS/rts-docs)), checked on the cluster on 2026-10-14 and updated with what
we learned while building the datasets (2026-10-22).
Items marked **[verify]** are not in the docs or are ambiguous: check them before relying on them.

**Contents**

1. Access
2. Hardware
3. Our resources
4. Slurm on Torch
5. Storage
6. Big datasets
7. Data transfers
8. OSN pod
9. Software environment
10. Multi-node DDP template
11. Misc
12. Open questions

## 1. ACCESS

- **Requirements**  : an *HPC account* (NetID, gives the login) **and** an *active allocation* in a project on
                      https://projects.hpc.nyu.edu (gives a Slurm account `torch_pr_XXX_XXXXX`).
                      No allocation → no jobs.
- **Collaborators** : a full-time NYU faculty sponsors an *affiliate* NetID, then the person requests the HPC account
                      (affiliation "External Collaborator"). Expires after **12 months** → yearly renewal.
- **VPN**           : **mandatory** off campus (login, OOD, project portal, status dashboard).
                      Linux: `sudo openconnect -b vpn.nyu.edu`.
- **Login**         : **SSH keys are NOT supported.** Password + Microsoft device-login 2FA
                      (https://microsoft.com/devicelogin + Duo) ⇒ no non-interactive SSH from outside, automation must
                      run *on* Torch.
- **VSCode**        : recommended `code-server` app in Open OnDemand (https://ood.torch.hpc.nyu.edu). Remote-SSH works
                      with `Remote.SSH: Connect Timeout = 120`, "Lockfiles In Tmp" on, "Use Flock" off, and
                      `~/.vscode-server` symlinked to `$SCRATCH/.vscode-server` (home inode quota). A reconnection
                      can land on another login node (`torch-login-a-0`, `a-1`, `b-1`, `b-3`, …): processes started
                      from the previous one keep running there (`ssh torch-login-b-3` from Torch to stop them).
- **Cluster usage** : https://stats.apps.cloud.rt.nyu.edu (VPN), or `vsinfo` (in `~/bin`).
- **Support**       : hpc@nyu.edu (personal consultations available, including DL scaling).

`~/.ssh/config` (from the docs):

```
Host torch login.torch.hpc.nyu.edu
  Hostname login.torch.hpc.nyu.edu
  User <NetID>
  StrictHostKeyChecking no
  ServerAliveInterval 60
  ForwardAgent yes
  UserKnownHostsFile /dev/null
  LogLevel ERROR
Host dtn.torch.hpc.nyu.edu
  User <NetID>
  (same options)
Host cs* cl* gr* ga* gh* gl*          # compute nodes, through the login node
  User <NetID>
  ProxyCommand ssh -W %h:%p torch-login
  ForwardAgent yes
```

---

## 2. HARDWARE

Infiniband NDR400 interconnect. Login: 4 nodes. DTN: 2 nodes (100 Gb/s to the research network).

| Type          | Prefix (`sinfo`)  | Nodes | GPUs / node | CPUs / node |    RAM / node | Total GPUs |
| ------------- | ----------------- | ----: | ----------: | ----------: | ------------: | ---------: |
| H200          | `gh1*`            |    34 |           8 |         128 |          2 TB |        272 |
| L40S          | `gl*`             |    68 |           4 |         128 |        512 GB |        272 |
| B200          | `gb*`             |    15 |           8 |         128 |          3 TB |        120 |
| RTX Pro 6000  | `gr*` (gr101-106) |     6 |           8 |         128 |        1.5 TB |         48 |
| H100          | `gh0*`            |    15 |           4 |          96 |        1.5 TB |         60 |
| A100          | `ga*`             |  9+34 |           4 |     64 / 80 |    0.5 / 1 TB |        172 |
| CPU std / big | `cs*` / `cl*`     | 186/8 |           – |         128 | 512 GB / 3 TB |          – |

- **RTX Pro 6000** = Blackwell (driver 610, sm_120) ⇒ PyTorch must be built for **CUDA ≥ 12.8**.
- General knowledge (not from the docs): RTX Pro 6000 Blackwell = 96 GB GDDR7, **no NVLink** (PCIe only) ⇒ intra-node
  all-reduce is slower than on H200/B200 SXM nodes.

---

## 3. OUR RESOURCES (Laure Zanna, M²LInES)

### Allocations

Project **"Zanna Research"** on https://projects.hpc.nyu.edu (verified 2026-10-14).

| Allocation        | Slurm account          | Reaches (no `--constraint`)                 | Ends                   |
| ----------------- | ---------------------- | ------------------------------------------- | ---------------------- |
| `torch` (Cluster) | `torch_pr_347_general` | CPUs, `l40s_public`, `h200_public`          | 2035-06-01             |
| `lzanna`          | `torch_pr_347_lzanna`  | `rtx6000_lzanna` + public GPUs              | 2027-07-14             |
| `courant`         | `torch_pr_347_courant` | `*_courant` (L40S, H200, RTX) + public GPUs | **2026-12-10** ⚠ renew |

CPU-only jobs land on the CPU partitions whatever the account ⇒ use `general` for them.

### Group caps

Partition QoS (`sacctmgr show qos`), shared by **every member** of the group.

| Partition         |  GPUs | CPUs |  Memory | Shared by                      |
| ----------------- | ----: | ---: | ------: | ------------------------------ |
| `rtx6000_lzanna`  | **8** |  128 |  1400 G | the whole lab (= one RTX node) |
| `rtx6000_courant` |     8 |  128 |  1400 G | all of Courant                 |
| `l40s_courant`    |    52 | 1664 |  6500 G | all of Courant                 |
| `h200_courant`    |    88 | 1408 | 22000 G | all of Courant                 |
| `l40s_public`     |   208 | 6656 | 26000 G | everyone                       |
| `h200_public`     |    24 |  384 |  6000 G | everyone                       |

⇒ A job of 8 RTX GPUs on lzanna takes **all** the lab's GPUs: keep big trainings on `courant` (H200, L40S), and
lzanna for jobs needing the RTX or a guaranteed slot.

### Choosing the account and the constraint

All verified with held jobs.

| Need                 | Flags                                                | Lands on                         |
| -------------------- | ---------------------------------------------------- | -------------------------------- |
| Lab RTX only         | `--account=torch_pr_347_lzanna --constraint=rtx6000` | `rtx6000_lzanna`                 |
| Courant H200         | `--account=torch_pr_347_courant --constraint=h200`   | `h200_courant` (+ `h200_public`) |
| Courant L40S         | `--account=torch_pr_347_courant --constraint=l40s`   | `l40s_courant` (+ `l40s_public`) |
| Any GPU we can reach | `--account=torch_pr_347_courant` (no constraint)     | first free partition             |
| CPU only             | `--account=torch_pr_347_general`                     | `cs`, `cpu_short`                |

---

## 4. SLURM ON TORCH

### Rules

- **`--account=<slurm_account>` is mandatory** on every `srun` / `sbatch`.
- **Do NOT set `--partition`.** Partitions are QoS-based views of stakeholder resources: Slurm dispatches to every
  partition the account reaches that matches the request. Pick the GPU type with `--constraint` (`rtx6000`, `h200`,
  `'h200|l40s'`). Features (`sinfo -o "%P %f"`): `rtx6000`, `l40s`, `h200`, `h100`, `a100`, `b200`, plus `cuda`.
- **`--mem=0` is refused** ("Please do not request memory 0"): request it explicitly, e.g. `--mem=1400G` per RTX node.
- **`--exclude=<node>` is refused** ("Please do note --exclude … report back to hpc@nyu.edu"): report a faulty node to
  hpc@nyu.edu instead.

### Limits per user

`show_slurm_qos`. The QoS follows the wall time, whatever the account, and the group caps (section 3) apply on top.

| QoS           | Wall time | Max per user          | Used when                       |
| ------------- | --------- | --------------------- | ------------------------------- |
| `interactive` | 6 h       | 16 CPUs, 60 GB        |                                 |
| `cpu_short`   | 6 h       | 32 CPUs, 120 GB       |                                 |
| `cpu48`       | 2 d       | 3000 CPUs, 6000 GB    | CPU jobs ≤ 48 h                 |
| `cpu168`      | 7 d       | 1000 CPUs, 2000 GB    | CPU jobs > 48 h                 |
| `cpuprem`     | 2 d       | 30000 CPUs (preempt.) |                                 |
| **`gpu48`**   | **2 d**   | **16 GPUs**           | GPU jobs ≤ 48 h                 |
| **`gpu168`**  | **7 d**   | **4 GPUs**            | GPU jobs > 48 h (sbatch notice) |

- The GPU caps are **per user, over all their running jobs** (`MaxTRESPerUser`, no per-job cap): one job can take the
  16 GPUs, or 2 jobs 8 each, the next ones waiting (`QOSMaxGRESPerUser`). At most 500 (`gpu48`) or 50 (`gpu168`) jobs
  per user in the queue (`sacctmgr show qos gpu48`, 2026-10-22).
- The docs also mention "24 GPUs per user for jobs < 48 h" (`QOSMaxGRESPerUser`), but `show_slurm_qos` gives
  `gres/gpu=16` for `gpu48` (2026-10-14).
- ⇒ Multi-node training: jobs ≤ 48 h, chained with checkpoint/restart.
- **At most 3000 jobs per user in the queue** (`MaxSubmitJobsPerUser` of `cpu48`), **each task of an array counting**
  (`QOSMaxSubmitJobPerUserLimit`) ⇒ submit big arrays in rolling batches (e.g. the next batch submitted by the last
  job of the current one).
- When the cluster is busy, Slurm starts the tasks of an array in slow batches (~15 min apart): a throttled array
  showing `JobArrayTaskLimit` with no task running is slow, not stuck.

### Cancellation of underused GPUs

Jobs with a low GPU utilization are cancelled automatically ("Enforcement will be very aggressive"). The delay before
the cancellation is not documented **[verify with hpc@nyu.edu]**.

| Nodes                                 | Cancelled below | Warned below |
| ------------------------------------- | --------------: | -----------: |
| `gh*` (H200)                          |            60 % |         75 % |
| `gl*`, `ga*`, `gr*` (L40S, A100, RTX) |            50 % |         70 % |
| other                                 |            10 % |         50 % |

⇒ Long CPU-only phases in GPU jobs (data staging, validation on CPU, big evaluations) can get the job killed: keep the
GPU busy and stage the data in a CPU job beforehand. Notebooks belong on CPUs.

### Preemption

| Partitions                                                                 | Preempted?                       |
| -------------------------------------------------------------------------- | -------------------------------- |
| `*_lzanna`, `*_courant`, `*_public`, `cs`, `cpu_short` (`PreemptMode=OFF`) | **never**                        |
| `h200`, `l40s`, `rtx6000`, … (reached only with `preemption=yes`)          | yes, after 30 min, then requeued |

- `#SBATCH --comment="preemption=yes;requeue=true"` → normal + preemptible partitions, to use idle stakeholder GPUs we
  don't own.
- `#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"` → preemptible only (may allow more
  resources).
- B200 (192 GB) are only reachable this way, e.g. `--account=torch_pr_347_courant --constraint=b200` with the comment above
  lands on `b200_courant_prem` (12 nodes × 8 B200, open to every account), handy for short memory tests (2026-10-25).
- Priority: stakeholder partitions > public partitions > preemptible partitions; GPU jobs preempt CPU jobs on GPU nodes.
- Requeued jobs restart from scratch ⇒ the training script must resume from its last checkpoint automatically.

### Pending reasons

| Reason                                                              | Meaning                                |
| ------------------------------------------------------------------- | -------------------------------------- |
| `QOSGrpGRES`                                                        | no free GPU in the partition (not you) |
| `QOSMaxGRESPerUser`, `QOSMaxMemoryPerUser`, `QOSMaxCpuPerUserLimit` | your personal limit                    |

### Misc

- Other `--comment` options (combinable with `;`): `gpu_mps=yes` (several jobs share a GPU), `ram_disk=1GB`
  (RAM-backed directory `$SLURM_RAM_TMPDIR`).
- Good citizenship: don't `watch squeue` / `sacct` (RPC throttling, check `sdiag`); use `--mail-type` and job arrays.
- Utilities: `my_slurm_accounts`, `myquota`, `rpsquota`, `seff <jobid>`, `show_slurm_qos`, `sdiag`, `nvidia-smi -l 5`,
  `/share/apps/images/run-nvtop-3.2.0.bash nvtop`.

---

## 5. STORAGE

| Path                 | Variable   | Quota (space / files) | Backup       | Purge             | Use                 |
| -------------------- | ---------- | --------------------- | ------------ | ----------------- | ------------------- |
| `/home/$USER`        | `$HOME`    | 50 GB / **30 K**      | daily        | no                | code, configs       |
| `/scratch/$USER`     | `$SCRATCH` | 5 TB / 5 M            | **no**       | **unused ≥ 60 d** | jobs, all-flash     |
| `/archive/$USER`     | `$ARCHIVE` | 2 TB / 20 K           | daily        | no                | tar, **login only** |
| RPS (`/projects/…`?) | –          | paid                  | daily (30 d) | no                | shared lab space    |

- Faking access times (`touch` scripts) to dodge the purge = **account locked**.
- **Home inode trap**: never put conda environments in `$HOME`; symlink `.vscode-server`, `.cursor-server` and conda
  `pkgs` to the scratch.
- RPS: $100/TB/yr + $100 per 200 K inodes/yr, at least 1 TB + 200 K inodes, requested by the PI by email. Check whether
  the Zanna group already has one (stakeholder list: Google sheet linked in the docs).
- Node-local disk: `$SLURM_TMPDIR`, `/state/partition1/$USER`.
- Sharing: NFSv4 ACLs (`nfs4_setfacl -a "A::<NetID>:RX" dir`, `-R -a "A:df:<NetID>:RWX" dir/share`), or an IPA Linux
  group `g-DEPT-LAB` (ask hpc@nyu.edu). Never `chmod 777`.
- High-risk data (PII, PHI, CUI) is forbidden on HPC.

---

## 6. BIG DATASETS

- **Avoid millions of small files** (inode quotas, metadata bottleneck). Prefer a few big files (Zarr with large chunks,
  HDF5, LMDB) or a **squashfs image** mounted read-only in Apptainer, several jobs being able to mount the same `.sqf`:
  ```bash
  mksquashfs mydata mydata.sqf -keep-as-directory      # build on node-local disk
  apptainer exec --overlay mydata.sqf:ro …             # data appears at /mydata
  ```
- Fastest reads: stage a subset to `$SLURM_TMPDIR` or a RAM disk at the start of the job (mind the GPU utilization
  killer while staging).
- **Internet on compute nodes**: the docs (PyTorch tutorial) say there is none, but `api.wandb.ai`, PyPI, GitHub and
  `nyu1.osn.mghpcc.org` were all reachable from compute nodes (2026-10-14).
- Public datasets (read-only): `/projects/work/public/ml-datasets/`.

---

## 7. DATA TRANSFERS

- **Never transfer on login nodes**: use the DTN `dtn.torch.hpc.nyu.edu` (rsync, scp, rclone) or **Globus** (endpoint
  **`nyu#torch`**, recommended for large volumes, automatic retries).
- `rclone` is available on the DTN and as a module (`module load rclone/1.72.1`, the only version on 2026-10-22).
- Open OnDemand upload: small files only.

---

## 8. OSN POD (Open Storage Network, S3-compatible object storage)

- **Endpoint**   : **`https://nyu1.osn.mghpcc.org`** (M²LInES pod).
- **Buckets**    : **`m2lines-pubs`**, public: anonymous listing and reads work since 2026-10-22 (refused before).
                   Others exist, e.g. `emulators` (seen in a Samudra PR).
- **Heimdall**   : datasets 2000-2025 in **`m2lines-pubs/Datasets`** (~12.4 TB, `README.md` at its root), paths in
                   `heimdall/config`, read anonymously by the code (no keys). Writing only with the rclone remote
                   **`osn-rw`** (`module load rclone/1.72.1`) or the s3fs profile **`osn`**, set up privately by each
                   user. Keys are never written in the repository nor displayed.
- **Access**     : *open* buckets = anonymous read, keys to write; *protected* buckets = keys to read and write, with
                   **two pairs, RO and RW** → use RO in training jobs.
- **Naming**     : lowercase, digits, `-_.`; no names differing only by case.
- **Keys, help** : https://coldfront.osn.mghpcc.org (keys), https://openstoragenetwork.github.io/docs/,
                   help@osn.mghpcc.org.

rclone remote (set up privately), then `rclone ls osn:/m2lines-pubs` or
`rclone copy --transfers=32 osn:/bucket/path $SCRATCH/path` (on the DTN):

```
[osn]
type = s3
provider = Ceph
access_key_id = <key>
secret_access_key = <secret>
endpoint = https://nyu1.osn.mghpcc.org
no_check_bucket = true
```

Python / xarray (public bucket):

```python
ds = xr.open_zarr("s3://m2lines-pubs/Datasets/Glorys/cmems_glorys_2000-2025_daily_4320x2041x50.zarr",
                  storage_options={"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org"}})
```

- **PyTorch DataLoader**: workers reading OSN through s3fs must be spawned (`multiprocessing_context="spawn"`), forked
  ones hang.
- **Throughput**: ~30 MB/s per S3 stream ⇒ parallelism comes from the dataloader workers, not the batch size; sharded
  zarr (one object per day) was ~2× faster to read than one object per chunk.
- **rclone**: `rclone about` is not supported by the pod, and `rclone purge` prints a harmless "Failed to read
  versioning status" error. Listings (`rclone lsf`, `rclone size --fast-list`) are fine; big copies go on the DTN or
  in a job.

---

## 9. SOFTWARE ENVIRONMENT (Apptainer + overlay)

**Heimdall**: `sbatch cluster/jobs/environment.sbatch` creates or updates everything.

| What    | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Overlay | `$SCRATCH/env/heimdall.ext3` (15 GB, 500 K files)              |
| Image   | `/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif` |
| Env     | conda environment `heimdall` (Miniforge), PyTorch `cu130`      |

- No Docker, Apptainer only (Docker images work: `apptainer pull docker://…`).
- **Install packages on a compute node**: login nodes cap memory at **2 GB/user** and run a *different OS* than compute
  nodes (`srun --account=<acct> -c4 --mem=16GB -t2:00:00 --pty bash`).
- **Jobs mount the overlay `:ro`**: a `:rw` mount locks the file, many jobs can share a `:ro` mount.
  Add `--nv` for GPUs.
- **Jupyter**: `sbatch cluster/jobs/jupyter.sbatch` (CPU node, add `--mem=96G` for a full day of GLORYS), then in
  VSCode *Select Kernel → Existing Jupyter Server* with `http://<node>:8888/lab?token=…` (node from `squeue`, token in
  `cluster/logs/jupyter.log`).
- `/tmp` is local to each node: files written there on a login node are not visible from the compute nodes.
- Remove any `conda init` block from `~/.bashrc` (it breaks overlay environments). Use `export PYTHONNOUSERSITE=True`.
- Images: `ls /share/apps/images/`. Overlay sizes: `ls /share/apps/overlay-fs-ext3/`.
- Optional: convert ext3 → squashfs (`mksquashfs /ext3 env.sqf -keep-as-directory`), ~2.5× smaller.
- Lmod modules exist (`module avail`), e.g. `anaconda3/2024.02`, `openmpi/gcc/4.1.6`, `rclone`.

Generic recipe (the environment lives in a single file ⇒ no inode problem):

```bash
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz $SCRATCH/env/ && gunzip $SCRATCH/env/overlay-15GB-500K.ext3.gz
apptainer exec --fakeroot \
    --overlay $SCRATCH/env/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda-13.3.1-ubuntu-26.04.sif bash
# inside: install Miniforge into /ext3/miniforge3 (or uv into /ext3/.uv + venv /ext3/.venv), write /ext3/env.sh
```

`uv` variant: `UV_INSTALL_DIR=/ext3/.uv`, `UV_PYTHON_INSTALL_DIR=/ext3/.uv/python`,
`UV_PROJECT_ENVIRONMENT=/ext3/.venv`, then `uv sync` from the repository with `pyproject.toml`.

---

## 10. MULTI-NODE DDP TEMPLATE (adapted from the docs)

Rules from the docs:

- tasks = GPUs: `--ntasks-per-node` = `--gres=gpu:N` (not `--ntasks`);
- fill whole nodes before going multi-node, never 1 GPU per node;
- NCCL backend, `local_rank = SLURM_PROCID % SLURM_GPUS_ON_NODE`;
- `gradient_as_bucket_view=True` saves ~1/3 of the memory;
- optimize single-GPU first (dataloader workers = `SLURM_CPUS_PER_TASK`).

Two full H200 nodes of Courant, i.e. the 16 GPUs allowed per user (lzanna, capped at 8 GPUs, cannot run it).

```bash
#!/bin/bash
#SBATCH --job-name=heimdall-ddp
#SBATCH --account=torch_pr_347_courant
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --constraint=h200
#SBATCH --cpus-per-task=14                        # 128 cores / 8 GPUs, keep a margin
#SBATCH --mem=1900G                               # --mem=0 is refused on Torch
#SBATCH --time=47:59:00
#SBATCH --comment="preemption=yes;requeue=true"   # optional, needs auto-resume
#SBATCH --output=cluster/logs/%x-%j.out

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun apptainer exec --nv \
    --overlay $SCRATCH/env/heimdall.ext3:ro \
    --overlay $SCRATCH/data/dataset.sqf:ro \
    /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
    bash -c "source /ext3/env.sh; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python train.py"
```

Job arrays share `SLURM_JOBID` ⇒ add `$SLURM_ARRAY_TASK_ID` to `MASTER_PORT`.

---

## 11. MISC

- Publications acknowledgment: *"This work was supported in part through the NYU IT High Performance Computing
  resources, services, and staff expertise"*; add the papers and grants to the project in the portal.
- Compared to Lucia (A100 40 GB, 4 per node, gpu QoS 48 GPUs per user): Torch's GPU QoS is much tighter per user
  (16 GPUs over 2 days, 4 GPUs over 7 days), and it cancels underused GPU jobs.

---

## 12. OPEN QUESTIONS

| #   | Question                                              | Status                                     |
| --- | ----------------------------------------------------- | ------------------------------------------ |
| 1   | OSN bucket of the project? RPS volume of the group?   | `m2lines-pubs`, public (§ 8); RPS **open** |
| 2   | Delay before an underused GPU job is cancelled?       | **open**, ask hpc@nyu.edu                  |
| 3   | `courant` allocation ends on **2026-12-10**           | **to do**, PI to renew it                  |
| 4   | lzanna's GPUs and group cap                           | 8 RTX for the whole lab (§ 3)              |
| 5   | `--constraint` names, node prefix ↔ GPU type          | see § 2 and § 4                            |
| 6   | Internet on compute nodes (OSN, WandB, PyPI, GitHub)? | yes, despite the docs (§ 6)                |
| 7   | Per-user GPU cap                                      | ≤ 48 h → 16, > 48 h → 4 (§ 4)              |
