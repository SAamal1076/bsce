# 16 Oct 2026 (v21): the workbench page showed an older workbench

**What you saw.** You started the v20 workbench. The page showed:
- the old steps, without the new 3GPP data task;
- the old runs;
- in Settings, the folder of v16.

**Why.**
- **The folder in Settings is the folder of the workbench that answers the browser.** So the page came from the v16 workbench, and not from v20. That happens in one of two ways:
  1. **The v16 workbench was still running** in its own window (the most likely case), and the v20 workbench could not take its port away.
     - The cause is Python's HTTP server: it sets the socket option SO_REUSEADDR. On Windows, that option lets a second program bind a port another program is still listening on.
     - So v20 bound port 8765 next to v16, instead of moving to the next free port as it would on Linux. The browser's connections to 8765 kept going to v16.
     - Every workbench up to v20 could do this.
  2. **The `start_app.bat` that was started belongs to the v16 folder** (a shortcut, for example). Its black window prints `bsce folder: …` on its second line.

**What v21 changes.**

| file | change |
|---|---|
| `app/bsce_app.py` | **Before it uses a port, the workbench checks whether something already answers there; if so, it takes the next port.** <br>• On Windows, it binds its port exclusively (SO_EXCLUSIVEADDRUSE, no SO_REUSEADDR), so two programs can no longer share a port. <br>• It recognises another bsce workbench: version and folder from v21 on, "an older version" before. The black window prints a NOTE with its address. <br>• New `/api/whoami`: answers app, version, folder and process id, without the key. <br>• New `/api/stop_other`: stops the bsce workbench on a given port, with its steps. <br>&nbsp;&nbsp;– It stops only a process that listens on that port and is `bsce_app.py`. <br>&nbsp;&nbsp;– Windows: `netstat` (language-independent), then the PowerShell command line, then `taskkill /T`. Linux: `/proc`. |
| `app/static/*` | **The page shows which workbench it is.** <br>• The version (`v21`) and the folder appear in the top bar, in the browser tab and in Settings. <br>• When another workbench is still running, a yellow note names it, with **Open it** and **Stop it** (with a confirmation that its running steps stop too). |
| `VERSION.txt` (new) | `21 (bsce_code_21, 16 Oct 2026)`. The workbench and `check_env.py` print it, so two bsce folders can be told apart. |
| `scripts/check_env.py` | Prints the version. |
| `tests/test_app.py` | **New test**, the situation you had. <br>• It starts an older workbench (this app without `/api/whoami`) in another folder. <br>• Then it starts the new one on the same port. <br>• **Checked:** <br>&nbsp;&nbsp;– the new one takes another port and prints the NOTE; <br>&nbsp;&nbsp;– its page data list the old one; <br>&nbsp;&nbsp;– "Stop it" stops it, and a second stop finds nothing. <br>• `probe_port` is also checked on a free port and on a port held by another program. |
| `docs/WORKBENCH.md`, `GUIDE.md` 8 | "Which workbench is this?" and "An older workbench is still open", and a row in the problems table. |

## What to do now (once)
1. **Close the black windows** of the old workbenches (v16 … v20).
   - Look first whether one is still training: closing a window stops its runs.
   - Their checkpoints stay in their own `outputs` folder.
2. **If you cannot find a window:**
   - In the Anaconda Prompt, run `netstat -ano | findstr :8765`. The last number of each line is a process id.
   - Then run `taskkill /PID <id> /T /F` for each id.
3. **Start v21:** double-click `start_app.bat` inside the v21 folder (not a shortcut to an older folder). The top bar must show `v21` and that folder.

From v21 on, this cannot happen again. A newer workbench takes its own port and shows the old one in the yellow note.

## Verification (this session, Linux)
- **A real v19 workbench running on 8765, then v21 started:**
  - v21 printed the NOTE ("an older version") and opened on 8766.
  - Its page showed the yellow note (screenshot).
  - "Stop it" (`/api/stop_other`) stopped the v19 process, and port 8765 was free again.
- **Tests:**
  - `tests/test_app.py`: 31 passed, with the new test.
  - The whole `tests/` folder on a clean copy of `bsce_code_21.zip` (Sionna and Octave + QuaDRiGa installed, TensorFlow not): 223 passed, 84 skipped, 0 failed.
- **Not tested here: the Windows-only parts.** These are SO_EXCLUSIVEADDRUSE, `netstat`, PowerShell and `taskkill`.
  - **The probe before binding:** it applies on every system, and it alone keeps a port that answers from being reused.
  - **`netstat` / PowerShell / `taskkill`:** they serve only the "Stop it" button. If they fail, the note says to close the old window.
