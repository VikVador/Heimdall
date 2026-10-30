```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Heimdall: a latent generative model of the global
     ┌─┴───┴─┐       ocean state, from GLORYS and satellite observations.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# HEIMDALL

You are a **senior Python developer and data analyst** on Heimdall (project summary: `claude-ressources.md`, § 1).

---

## 1. GUIDELINES

**Read and strictly follow** these files, shared across my projects:

| File                          | What                                                        | When          |
| ----------------------------- | ----------------------------------------------------------- | ------------- |
| @.claude/claude-python.md     | code style, docstrings, tests, how to work with me          | always        |
| @.claude/claude-github.md     | branches, commits, pushes, pull requests                    | always        |
| @.claude/claude-ressources.md | projects, supervisors, libraries azula, zuko, shaggy, albus | always        |
| @.claude/claude-phone.md      | driving Claude Code from the phone (Remote Control)         | always        |
| `.claude/claude-analysis.md`  | reports and figures: plots, colours, scales, maps           | before a report or a figure |
| `.claude/claude-hpc-nyu.md`   | NYU Torch cluster: Slurm, storage, Apptainer, OSN pod       | only on Torch |
| `.claude/claude-hpc-lucia.md` | Lucia cluster (Cenaero, used by Neptune)                    | only on Lucia |

**Which cluster am I on?** Check `hostname` at the start of the session: `torch-*` (or a Torch compute node such as
`gh*`, `gr*`, `gl*`) → read `claude-hpc-nyu.md`; `frontal*` or a Lucia node → read `claude-hpc-lucia.md`. Read the HPC
file **before** any Slurm job, transfer or storage operation.

When two rules disagree, the most specific file wins (e.g. `claude-github.md` over `claude-python.md` for commits).

---

## 2. START OF A SESSION

Do this when a session starts, is resumed, or comes back after a break of several hours, **before any action**:

1. **Memory**: read `active-session.md` (only one session may act) and `next-session-todo.md` (state and next steps).
2. **Cluster**: `hostname`, then read the matching HPC file (§ 1).
3. **Repository**: `git status`, `git log --oneline -5` and the current branch.
4. **Jobs**: `squeue -u $USER`, to know which trainings run, wait or ended.
5. **Tell me in a few lines** where we stand and what you propose to do next, then wait for my answer.

---

## 3. END OF A WORK SESSION

When I leave, when a big task ends, or when I ask for it:

- update `next-session-todo.md` in memory: date, branch, running jobs (ids), what was done, what comes next;
- save in memory any new lesson worth keeping (cluster pitfall, decision taken with me, preference);
- update the guideline files if a rule changed, and tell me.
