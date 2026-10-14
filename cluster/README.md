```
  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
       ~~~      .-'''-.      ~~~       .-'''-.      ~~~       .-'''-.
  _.-'''-._.-'''       '''-._.-'''-._.'       '''-._.-'''-._.'       '''-.


                      H   E   I   M   D   A   L   L


  ~~  ~~~  ~~~   ~~  ~~~~  ~~   ~~~  ~~~   ~~  ~~~  ~~  ~~  ~~~  ~~~   ~~
 ~~~~   ~~   ~~~~   ~~~   ~~   ~~~~   ~~   ~~~~   ~~~  ~~ ~~~~   ~~   ~~~~
```

Never set `--partition`: the account and the constraint select it. On the partitions below, a job is **never preempted**,
it only stops when its `--time` is reached or when its GPUs stay underused.

# CPU

| Account                | Constraint | CPUs / node | `--mem` (full node) | Group cap                   |
|------------------------|------------|------------:|--------------------:|-----------------------------|
| `torch_pr_347_general` | –          | 128         | `480G`              | None (limits per user only) |

# GPU

| Account                | Constraint | CPUs / node | `--mem` (full node) | Group cap               | GPU          | GPUs / node | Memory / GPU | CPUs / GPU |
|------------------------|------------|------------:|--------------------:|-------------------------|--------------|------------:|-------------:|-----------:|
| `torch_pr_347_courant` | `h200`     | 128         | `1900G`             | 88 GPUs (all Courant)   | H200         | 8           | 141 GB       | 16         |
| `torch_pr_347_courant` | `l40s`     | 128         | `480G`              | 52 GPUs (all Courant)   | L40S         | 4           | 48 GB        | 32         |
| `torch_pr_347_courant` | `rtx6000`  | 128         | `1400G`             | 8 GPUs (all Courant)    | RTX Pro 6000 | 8           | 96 GB        | 16         |
| `torch_pr_347_lzanna`  | `rtx6000`  | 128         | `1400G`             | **8 GPUs (whole lab)**  | RTX Pro 6000 | 8           | 96 GB        | 16         |

Note: H200 and L40S are also reachable with `torch_pr_347_general` (public, 24 and 208 GPUs for the whole cluster).

# Limits per user

| `--time` | GPUs    | CPUs  |
|----------|--------:|------:|
| ≤ 48 h   | **16**  | 3000  |
| ≤ 7 days | **4**   | 1000  |

# Cancellation

- GPU utilization below **50 %** (L40S, RTX Pro 6000) or **60 %** (H200).
- `--time` reached.

# Examples

**1 GPU**, its share of an H200 node.

```
#SBATCH --account=torch_pr_347_courant
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=237G
#SBATCH --time=48:00:00
```

**8 GPUs**, a full H200 node, one process per GPU.

```
#SBATCH --account=torch_pr_347_courant
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=1900G
#SBATCH --time=48:00:00
```

**16 GPUs**, the maximum per user, two full H200 nodes (`--gres` and `--mem` being per node).

```
#SBATCH --account=torch_pr_347_courant
#SBATCH --constraint=h200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=1900G
#SBATCH --time=48:00:00
```
