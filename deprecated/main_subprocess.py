import os
import sys
import subprocess

if __name__ == '__main__':
    worker1_read, worker2_write = os.pipe()
    worker2_read, worker1_write = os.pipe()

    cmd1 = [sys.executable, "-m", "./deit/main", str(worker1_read),
            str(worker1_write)]
    cmd2 = [sys.executable, "-m", "./rl/actor_critic", str(worker2_read),
            str(worker2_write)]

    proc1 = subprocess.Popen(cmd1, pass_fds=(worker1_read, worker1_write))
    proc2 = subprocess.Popen(cmd2, pass_fds=(worker2_read, worker2_write))

    


