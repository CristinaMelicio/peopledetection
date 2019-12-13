import sched
import time
import subprocess
from datetime import datetime


INTERVAL_LD = 1
SEQ_I = 234
SEQ_F = 300


def run_script(scheduler):
    for f_id in range(SEQ_I, SEQ_F):
        for cam_id in [1, 2]:
            print("In", datetime.now())
            subprocess.call(['python', 'seq_extract.py'] +
                            [str(f_id), str(cam_id)])
            print("Out", datetime.now())
            scheduler.enter(INTERVAL_LD,
                            1,
                            run_script,
                            (scheduler, ))


def main():
    s = sched.scheduler(time.time, time.sleep)
    s.enter(1, 1, run_script, (s,))
    s.run()


if __name__ == "__main__":
    main()
