```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Phone: drive Claude Code from the phone while it
     ┌─┴───┴─┐       works on a cluster (Remote Control in tmux).
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# CLAUDE PHONE

**Set up: 2026-10-26** on `torch-login-a-1`.

Drive Claude Code from the phone (Claude mobile app, or https://claude.ai/code) while it works on Torch, through the
Remote Control of Claude Code.

**Contents**

0. In short
1. Usage
2. How it works
3. How long it stays open
4. Requirements
5. Phone answer mode
6. Install on a new machine

---

## 0. IN SHORT

1. **Start** in a VSCode terminal on Torch: `claude-phone`. **Leave** without stopping it: **Ctrl+B**, then **D**.
2. **Phone**: Claude app → **Code** tab → session **heimdall**.
3. **One active session at a time**: never give work to the phone session and to VSCode simultaneously.
4. **tmux is per node**: check or stop the session from the node where it runs (`torch-login-a-1`).
5. **Phone answer mode**: short answers for a small screen, turned on with `/phone on`, off with `/phone off` (§ 5).

---

## 1. USAGE

| Command               | Effect                                                                   |
| --------------------- | ------------------------------------------------------------------------ |
| `claude-phone`        | Starts the tmux session `claude` if needed, then attaches to it          |
| `claude-phone stop`   | Kills the tmux session (and Claude with it)                              |
| `claude-phone status` | Tells whether the session runs **on the current node** (prints its name) |

| Step | Where           | What                                                                            |
| ---- | --------------- | ------------------------------------------------------------------------------- |
| 1    | VSCode on Torch | `claude-phone`                                                                  |
| 2    | same terminal   | leave without stopping Claude: **Ctrl+B**, then **D** (or close the terminal 🗑️) |
| 3    | phone           | Claude app → **Code** tab → session **heimdall**                                |

| Option (environment variable) | Effect                                         | Default       |
| ----------------------------- | ---------------------------------------------- | ------------- |
| `CLAUDE_PHONE_MODE`           | permission mode (`default`, `plan`, `auto`, …) | `acceptEdits` |
| `CLAUDE_PHONE_DIR`            | working directory                              | `~/heimdall`  |

---

## 2. HOW IT WORKS

| Piece          | Details                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Script         | `~/bin/claude-phone` (in `PATH`, outside the repository)                                              |
| Claude binary  | `claude` from `PATH` (standalone CLI in `~/.local/bin`), otherwise the latest one bundled with VSCode |
| Command        | `claude remote-control` in a detached tmux session, from `~/heimdall`                                 |
| Login node use | `nice -n 10`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` set to 4 (shared nodes) |
| Network        | outbound HTTPS to Anthropic: the phone needs neither the NYU VPN nor the 2FA                          |

**Restart loop**

- The first run creates the session `heimdall` (`--name`), the next ones reattach to it (`-c`).
- After a failure, it restarts 30 s later.
- It stops on exit code 0 or 130 (Ctrl+C); Ctrl+C during the 30 s wait also stops it.

---

## 3. HOW LONG IT STAYS OPEN

| Event                                       | Consequence                                                     |
| ------------------------------------------- | --------------------------------------------------------------- |
| Network outage of about 10 minutes          | Claude exits, the loop reattaches with `-c` every 30 s          |
| `-c` only reattaches within about **4 h**   | After 480 failed retries the loop gives up: `stop` then `start` |
| Reboot of the login node                    | Everything is lost: run `claude-phone` again                    |
| Long processes killed by the cluster admins | Same, run `claude-phone` again                                  |
| `claude-phone stop`, or exit in the session | Ends normally                                                   |

**Good to know**

- **tmux is per node.** VSCode can reconnect to another login node (`torch-login-a-0`, `a-1`, `b-1`, `b-3`, …):
  `claude-phone status` then says "not running". Go back with `ssh torch-login-a-1` before `claude-phone stop` or
  attaching, otherwise you start a **second** session.
- **Login nodes are for light work.** The NYU docs do not forbid long, light processes there, but heavy work
  (training, analysis, transfers) goes through Slurm jobs or the DTN. Memory is capped at 2 GB per user (Claude uses
  ~350 MB). Ask hpc@nyu.edu in case of doubt.

---

## 4. REQUIREMENTS

| Requirement     | Details                                                                       |
| --------------- | ----------------------------------------------------------------------------- |
| Claude Code CLI | ≥ 2.1, installed with `curl -fsSL https://claude.ai/install.sh \| bash`       |
| Login           | a **claude.ai account** (`/login`), API keys are not supported                |
| Remote Control  | allowed by the **personal** account, blocked by the NYU-managed one           |
| Phone           | logged in with the **same account** as the CLI                                |
| tmux            | `/usr/bin/tmux` on Torch                                                      |
| One session     | **one active session at a time**: never the phone session and VSCode together |

The NYU-managed account shows the block in `~/.claude/policy-limits.json` (`allow_remote_control: false`).

---

## 5. PHONE ANSWER MODE

In the session started by `claude-phone`, answers can switch to a **phone style**, short enough for a small screen.

| When                                        | What happens                                       |
| ------------------------------------------- | -------------------------------------------------- |
| First prompt of the session                 | Claude first asks: "Es-tu sur ton téléphone ?"     |
| More than **2 h** since the previous prompt | Same question                                      |
| Answer yes / no                             | Claude turns the mode on / off by itself           |
| `/phone on`, `/phone off`, `/phone`         | Turns it on, off, or toggles it, at any time       |
| Mode on                                     | Every prompt reminds Claude to use the phone style |

**Phone style**: every answer has three parts, separated by a horizontal rule (a `---` line between blank lines).

| Part       | Content                                                                            |
| ---------- | ---------------------------------------------------------------------------------- |
| 1. Title   | the task in a few words, in **bold**                                               |
| 2. Summary | 2 to 4 lines of plain text: what was done or found, and why, very simple and clear |
| 3. Changes | one bullet per change: *what* changed, then *why* in one short, simple sentence    |

- Part 3 is left out when nothing was changed (a simple question, a status check).
- A question to me comes last, on its own line, with the expected answers: `👉 Je relance ? (oui / non)`.
- No wide tables, no code blocks, no long paths: file names only. I can ask "détaille" for the normal answer.

Example:

```markdown
**SSS : valeurs aberrantes retirées**

---

Les pics de la validation venaient de quelques salinités impossibles (> 45 PSU)
en 2019-2021. Je les ai retirées et l'entraînement repart de zéro (≈ 20 h).

---

- `data/__init__.py` : les salinités > 45 PSU deviennent des trous (NaN),
  comme les zones sans mesure.
- `encoder-sss.yml` : relancé de zéro, car les anciens poids avaient appris ces pics.
- Job 21012345 : en attente d'une RTX du labo, démarrage estimé ce soir.

👉 Je pousse la branche ? (oui / non)
```

**How it works** (outside the repository, for all projects)

| Piece                             | Role                                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| `CLAUDE_PHONE=1`                  | set by `claude-phone` in the tmux session: the hooks act only there |
| `~/.claude/hooks/phone-prompt.sh` | hook on every prompt: 2 h check, reminder when the mode is on       |
| `~/.claude/hooks/phone-start.sh`  | hook at startup: the first prompt of a new session always asks      |
| `~/.claude/hooks/phone-mode.sh`   | turns the mode on / off (state file `~/.claude/phone/mode`)         |
| `~/.claude/commands/phone.md`     | the `/phone` command                                                |
| `~/.claude/settings.json`         | declares the two hooks                                              |

---

## 6. INSTALL ON A NEW MACHINE

The scripts live in `.claude/phone/` of the repository; copy them to the places Claude Code and the shell expect.

| From `.claude/phone/`                                | To                                                         |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| `claude-phone`                                       | `~/bin/claude-phone` (in `PATH`)                           |
| `phone-mode.sh`, `phone-prompt.sh`, `phone-start.sh` | `~/.claude/hooks/`                                         |
| `phone.md`                                           | `~/.claude/commands/phone.md`                              |
| `settings-hooks.json`                                | merged into `~/.claude/settings.json` (keep what is there) |

```bash
mkdir -p ~/bin ~/.claude/hooks ~/.claude/commands
cp .claude/phone/claude-phone ~/bin/ && cp .claude/phone/phone-*.sh ~/.claude/hooks/
cp .claude/phone/phone.md ~/.claude/commands/ && chmod +x ~/bin/claude-phone ~/.claude/hooks/phone-*.sh
```

- Set `CLAUDE_PHONE_DIR` (or edit the default `~/heimdall` in the script) to the repository of the project.
- After a change of a script, copy it back into `.claude/phone/` so that both stay identical.
