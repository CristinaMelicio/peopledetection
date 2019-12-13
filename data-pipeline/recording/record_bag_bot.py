import sched
import time
import subprocess
from datetime import datetime


INTERVAL_LD = 1

def run_script(scheduler):
    time_now = datetime.now().time()
    h = time_now.strftime("%H")
    m = time_now.strftime("%m")
    hour_now = float(h) + float(m) / 60

    if (hour_now > 11.0 and
        hour_now < 14.0) or (
            hour_now > 16.0 and
            hour_now < 18.0):

        print("In", datetime.now())
        subprocess.call(['python', 'record_bot.py'])
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
