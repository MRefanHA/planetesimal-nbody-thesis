# Part 1 — VS Code setup, step by step

Follow this in order. Do not skip ahead. Each step takes 1–5 minutes.

If anything fails, run `python check_setup.py` — it tells you what is wrong
and the exact command that fixes it.

---

## Step 0 — Clean up the mess first

Your VS Code has been crashing. Before adding anything new, close it and get
rid of whatever half-broken state is there.

1. Close VS Code completely.
2. If you already have a `planetesimal_sim_thesis` folder somewhere, **rename
   it** to `planetesimal_sim_thesis_OLD`. Don't delete it yet — just move it
   out of the way so you start clean.

---

## Step 1 — Build the folder correctly

This is where most of the trouble comes from. You downloaded files one at a
time, so you have to rebuild the folder structure by hand. It must look
**exactly** like this:

```
planetesimal_sim_thesis/
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── planetesimal_sim/
│   ├── __init__.py          <- easy to forget, nothing works without it
│   ├── units.py
│   ├── config.py
│   ├── orbits.py
│   ├── ic.py
│   ├── forces.py
│   ├── collisions.py
│   ├── integrator.py
│   └── analysis.py
├── tests/
│   └── test_validation.py   <- must be INSIDE tests/, not in the root
├── benchmarks.py
├── check_setup.py
├── run_scenarios.py
├── requirements.txt
├── README.md
├── planetesimal_sim_colab.ipynb
├── CHAPTER_I_II_corrections.md
└── SECTION_2_3_amendments.md
```

Three things that go wrong here:

- **`__init__.py` is missing.** It is a small file and easy to overlook.
  Without it, `from planetesimal_sim import SimConfig` fails.
- **`test_validation.py` sits in the root instead of `tests/`.** That file
  works out its import path by going up two folder levels, so it only works
  from inside `tests/`. This is exactly the error you hit on Colab.
- **`.vscode` is invisible on Windows.** Folders starting with a dot are
  hidden by default. In File Explorer: View → Show → Hidden items.

### Where to put the folder

Put it somewhere simple, like:

```
C:\thesis\planetesimal_sim_thesis        (Windows)
/Users/yourname/thesis/planetesimal_sim_thesis   (Mac)
```

**Do not put it inside a Google Drive or OneDrive synced folder.** The sync
client and VS Code's file watcher will both try to index every simulation
output file as it is written, and they will fight each other. That alone can
freeze the editor. Keep your working copy local; copy results to Drive later
if you want a backup.

Avoid spaces and non-ASCII characters in the path if you can. `C:\thesis\` is
safer than `C:\Users\Rey\Documents\Tugas Akhir\Simulasi\`.

---

## Step 2 — Check Python is installed

Open a terminal (Windows: press Start, type `cmd`, Enter. Mac: open Terminal)
and run:

```bash
python --version
```

You want **3.10, 3.11, or 3.12**. If it says 3.13 or newer, numba may not
support it yet and `pip install pytreegrav` will fail — install 3.12 from
python.org alongside it.

If you get "python is not recognized", Python isn't installed or isn't on your
PATH. Install from python.org and **tick "Add Python to PATH"** in the
installer. On Mac, try `python3 --version` instead.

---

## Step 3 — Open the folder in VS Code

VS Code → **File → Open Folder** → select `planetesimal_sim_thesis`.

**Open the folder, not individual files.** If you open a single `.py` file,
VS Code has no idea where the project root is and nothing will import. This
is the single most common beginner mistake with Python in VS Code.

When it asks "Do you trust the authors of the files in this folder?", say yes.

You should see the file tree in the left sidebar, matching Step 1.

---

## Step 4 — Install the Python extension

Left sidebar → Extensions icon (four squares) → search `Python` → install the
one by **Microsoft**. It brings Pylance with it. That is all you need — do not
install a pile of other extensions right now.

---

## Step 5 — Create a virtual environment

This keeps the project's packages separate from the rest of your system. Two
commands, and it prevents a whole category of "it worked yesterday" problems.

Open the terminal inside VS Code: **Terminal → New Terminal** (or ``Ctrl+` ``).
Make sure the prompt shows your project folder. Then:

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

The prompt should now start with `(.venv)`. If Windows refuses with an
execution-policy error, run this once in PowerShell and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Step 6 — Install the packages

With `(.venv)` showing in the prompt:

```bash
pip install -r requirements.txt
```

This installs numpy, scipy, matplotlib and pytreegrav (which pulls in numba).
It takes a couple of minutes. numba is a big download — that is normal.

---

## Step 7 — Point VS Code at the right Python

Press `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) → type **"Python: Select
Interpreter"** → choose the one with `.venv` in its path. It is usually at the
top, marked "Recommended".

Then `Ctrl+Shift+P` → **"Developer: Reload Window"**. This makes VS Code pick
up the new interpreter and the `.vscode/settings.json` file.

---

## Step 8 — Run the setup checker

```bash
python check_setup.py
```

This is the moment of truth. It checks your Python version, verifies every
file is where it should be, confirms the packages are installed, imports the
package, and runs a two-second simulation to prove the physics works.

Expected ending:

```
======================================================================
Setup is good.

Next steps, in order:
  1.  python tests/test_validation.py     (~1 min)
  2.  python benchmarks.py --quick        (~40 s)
  3.  python benchmarks.py                (~6 min, full B4 table)
  4.  python run_scenarios.py --all --dry-run
======================================================================
```

If it fails, it prints exactly what is missing and the command that fixes it.
Fix one thing, run it again, repeat.

It will also print a rough estimate of how long a production run will take on
**your** machine. On the reference machine (1 core) that came out at ~8 h per
scenario, ~33 h for all four. Your number will differ — that is the point of
measuring it.

---

## Step 9 — Run the validation tests

```bash
python tests/test_validation.py
```

All eight should pass in about a minute. This is what proves the code is
physically correct, not just that it runs. Keep this output — it belongs in
your thesis appendix.

---

## Step 10 — Learn the code with the debugger

This is the step that will actually make you understand what you're running,
and it is worth an hour of your time before you talk to your supervisors.

1. Open `planetesimal_sim/integrator.py`.
2. Find the line `def step(self, dt):` (around line 60).
3. Click in the narrow margin just left of the line number on the first line
   *inside* that function. A red dot appears. That is a breakpoint.
4. Press **F5**. Choose **"TINY run - step through this one to learn the
   code"** from the dropdown. It runs 200 particles for 5 years — small enough
   to step through.
5. Execution stops at your red dot. Now:
   - **F10** steps to the next line
   - The **Variables** panel on the left shows every value live
   - Hover over any variable in the code to see its current value

Walk through one complete timestep with F10. You will see, in order:

1. `_accel()` computes the gravity — star, then tree, then planet, then drag
2. `st["vel"] += 0.5*dt*a_p` — the first half-kick
3. `resolve_collisions(...)` — the collision search
4. `st["pos"] += dt*st["vel"]` — the drift
5. `_accel()` again — recompute forces at the new positions
6. the second half-kick

That sequence *is* your method. Once you have watched it happen once, you can
describe it to your professors from memory.

Try this while you're in there: hover over `st["pos"]` and look at the shape.
It is `(200, 3)` — 200 particles, 3 coordinates each. That single fact makes
most of the rest of the code readable.

---

## If VS Code still crashes

Check these in order:

1. **Is `.vscode/settings.json` actually there?** Run
   `python check_setup.py` — it warns if the file is missing. That file is
   what stops VS Code indexing hundreds of `.npz` output files.
2. **Are you running long jobs in the integrated terminal?** Don't. For
   anything over a few minutes, use a separate system terminal window.
   Thousands of printed lines sitting in VS Code's buffer will choke it.
3. **Is the folder inside Google Drive / OneDrive?** Move it out.
4. **Is `results/` already full of files from a crashed run?** Delete the
   whole `results/` folder and start again.
5. **Turn off Auto Save** (File menu → Auto Save, untick) while a simulation
   is running.

---

## Quick reference

| I want to... | Command |
|---|---|
| Check everything is set up | `python check_setup.py` |
| Prove the physics is right | `python tests/test_validation.py` |
| Get my Chapter III tables | `python benchmarks.py` |
| See the four scenarios' parameters | `python run_scenarios.py --all --dry-run` |
| Run a tiny test | `python run_scenarios.py --scenario S1_gravity_only --n 200 --t-end 10 --outdir /tmp/test` |
| Run production | `python run_scenarios.py --all` |
| Make figures | `python -m planetesimal_sim.analysis results` |

Every command runs from the project root, with `(.venv)` active.
