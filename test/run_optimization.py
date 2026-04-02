from pathlib import Path
import subprocess
import json
import random
import numpy as np
import time

project_dir = Path(__file__).resolve().parent
JULIA_CMD = ["julia", "run_via_python.jl"]
RUNCASE_TIMEOUT = 200.0


def start_node():
    return subprocess.Popen(
        JULIA_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=project_dir,
        bufsize=1,
    )


def stop_node(p):
    if p is None:
        return

    try:
        if p.stdin and not p.stdin.closed:
            try:
                # polite shutdown message, if the node is still alive
                term_msg = json.dumps([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "None", True])
                p.stdin.write(term_msg + "\n")
                p.stdin.flush()
            except Exception:
                pass

        p.terminate()
        p.wait(timeout=5)
    except Exception:
        try:
            p.kill()
            p.wait(timeout=5)
        except Exception:
            pass


proc = start_node()


def restart_node():
    global proc
    print("Restarting Julia node...")
    stop_node(proc)
    proc = start_node()


def get_log_file(case_name: str) -> Path:
    return project_dir / "test" / "evaluation" / "data" / case_name / "logs.txt"


def log_has_segfault(case_name: str) -> bool:
    log_file = get_log_file(case_name)
    if not log_file.exists():
        return False

    try:
        text = log_file.read_text(errors="ignore")
    except Exception:
        return False

    return ("Segmentation fault" in text) or ("signal 11" in text)


def read_one_line_with_timeout(p: subprocess.Popen, timeout: float):
    start = time.time()

    while True:
        # process already died
        if p.poll() is not None:
            try:
                remaining = p.stdout.read() if p.stdout else ""
            except Exception:
                remaining = ""
            raise RuntimeError(f"Julia node exited unexpectedly. Remaining output:\n{remaining}")

        # try to read a line if available
        # readline() is blocking, so do it only after checking timeout window carefully
        if time.time() - start > timeout:
            raise TimeoutError(f"run_case exceeded {timeout} seconds")

        # small sleep to avoid busy loop
        time.sleep(0.1)

        # use a short non-blocking-ish probe by relying on line buffering behavior
        # if Julia flushes lines, this will return once a line is complete
        if p.stdout is not None:
            try:
                line = p.stdout.readline()
                if line:
                    return line.strip()
            except Exception as e:
                raise RuntimeError(f"Failed reading Julia output: {e}") from e


def run_case(x0, y0, z0, yaw0, xf, yf, zf, yawf, name, finished=False):
    global proc

    msg = json.dumps([x0, y0, z0, yaw0, xf, yf, zf, yawf, name, finished])

    # if process already died before sending, restart it
    if proc.poll() is not None:
        restart_node()

    try:
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()
    except Exception:
        restart_node()
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()

    try:
        line = read_one_line_with_timeout(proc, RUNCASE_TIMEOUT)
        return line

    except TimeoutError:
        print(f"Timeout for case '{name}'. Checking logs...")

        if log_has_segfault(name):
            print(f"Segmentation fault detected in logs for '{name}'.")
            restart_node()
            return "SEGFAULT_RESTARTED"

        raise TimeoutError(
            f"Case '{name}' exceeded {RUNCASE_TIMEOUT} seconds, "
            f"but no segmentation fault was found in {get_log_file(name)}"
        )

    except RuntimeError as e:
        print(f"Julia node died during case '{name}': {e}")

        if log_has_segfault(name):
            print(f"Segmentation fault detected in logs for '{name}'.")
            restart_node()
            return "SEGFAULT_RESTARTED"

        restart_node()
        raise


size_x = 5
center_x = -3
size_y = 20
center_y = 0
size_z = 10
center_z = -5.5

kyaw = [-1, 0, 1]

try:
    for i in range(20, 30):
        x0 = random.random() * size_x - size_x / 2 + center_x
        y0 = random.random() * size_y - size_y / 2 + center_y
        z0 = random.random() * size_z - size_z / 2 + center_z
        yaw0 = random.random() * 2 * np.pi - np.pi

        xf = random.random() * size_x - size_x / 2 + center_x
        yf = random.random() * size_y - size_y / 2 + center_y
        zf = random.random() * size_z - size_z / 2 + center_z
        yawf = random.random() * 2 * np.pi - np.pi

        for j in range(0, 3):
            yawfinal = yawf + 2 * np.pi * kyaw[j]
            name_i = f"case{i*3+j}"

            res = run_case(x0, y0, z0, yaw0, xf, yf, zf, yawfinal, name_i, False)
            print(f"Result of {name_i}: {res}")

finally:
    try:
        run_case(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "None", True)
    except Exception:
        pass
    stop_node(proc)