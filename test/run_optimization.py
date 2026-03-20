from pathlib import Path
import subprocess
import json
import random
import numpy as np

project_dir = Path(__file__).resolve().parent

proc = subprocess.Popen(
    ["julia", "run_via_python.jl"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    cwd=project_dir
)

def run_case(x0, y0, z0, yaw0, xf, yf, zf, yawf, name, finished=False):
    msg = json.dumps([x0, y0, z0, yaw0, xf, yf, zf, yawf, name, finished])
    proc.stdin.write(msg + "\n")
    proc.stdin.flush()

    line = proc.stdout.readline().strip()
    return line


# res1 = run_case(0.0, 0.0, 5.0, 5.0, "case1", False)
# print("Result 1:", res1)

# res2 = run_case(1.0, 2.0, 6.0, 7.0, "case2", False)
# print("Result 2:", res2)

# res3 = run_case(-2.0, 1.0, 3.0, 4.0, "case3", False)
# print("Result 3:", res3)

# res4 = run_case(-20.0, 1.0, 15.0, -10.0, "case4", False)
# print("Result 4:", res4)

# res5 = run_case(10.0, 10.0, 20.0, 25.0, "case5", False)
# print("Result 5:", res5)

# res6 = run_case(-25.0, -10.0, -27.0, -50.0, "case6", False)
# print("Result 6:", res6)

# res7 = run_case(-25.0, -10.0, -27.0, 10.0, "case7", False)
# print("Result 7:", res7)

# termination = run_case(-25.0, -10.0, -27.0, 10.0, "case8", True)
# print("Result 8:", termination)

size_x = 20
center_x = 0
size_y = 20
center_y = 0
size_z = 10
center_z = -5.5


kyaw = [-1,0,1]
for i in range(0,15):
    x0 = random.random() * size_x - size_x/2 + center_x
    y0 = random.random() * size_y - size_y/2 + center_y
    z0 = random.random() * size_z - size_z/2 + center_z
    yaw0 = random.random() * 2*np.pi - np.pi
    xf = random.random() * size_x - size_x/2 + center_x
    yf = random.random() * size_y - size_y/2 + center_y
    zf = random.random() * size_z - size_z/2 + center_z
    yawf = random.random() * 2*np.pi - np.pi
    for j in range(0,3):
        yawfinal = yawf + 2*np.pi*kyaw[j]
        name_i = f"case{i*3+j}"
        res = run_case(x0, y0, z0, yaw0, xf, yf, zf, yawfinal, name_i, False)
        print(f"Result of {name_i}: {res}")

run_case(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "None", True)