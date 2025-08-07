"""
Runs the g4beamline shell script
"""

import subprocess
import os

def run_g4bl(sh_path):
    cmd = "echo 'hello'"
    subprocess.run(["echo", "hello world"], check=True, shell=True)
    subprocess.run(
        ["C:/cygwin/bin/bash.exe",
         "-l",
         "-c",
         "echo 'hello world'"],
         check=True,
         shell=True
    )

if __name__ == '__main__':
    # sh_path = "'C:/Users/Timothy Chew/Desktop/UROP2025/MakingMatterfromLight2025/g4bl _stuff/test.sh'"
    # print(        ["C:/cygwin/bin/bash.exe",
    #      "-l",
    #      "-c",
    #      f"bash '{sh_path}'"])
    run_g4bl(
        sh_path = "'C:/Users/Timothy Chew/Desktop/UROP2025/MakingMatterfromLight2025/g4bl_stuff/test.sh'"
    )