"""Init both arms in position-holding mode and watch shoulder drift for 10s.

    /home/andon/yam-tests/i2rt/.venv/bin/python /home/andon/yam-tests/molmoact2-setup/scripts/hold_test.py

Shoulder pitch (q[1]) should change by < 0.01 rad total. If left drifts >0.05
rad while right stays put, the kp=0 fix wasn't the (whole) issue.
"""
import sys, time
sys.path.insert(0, "/home/andon/yam-tests/molmoact2-setup/scripts")
from yam_client import init_arm

print("initing can0...")
l = init_arm("can0", "linear_4310")
print("initing can1...")
r = init_arm("can1", "linear_4310")
print("\nholding 10s -- shoulder pitch (q[1]) should NOT drift:")
for i in range(10):
    print(f"  t+{i+1}s  left q[1]={l.get_joint_pos()[1]:+.3f}  right q[1]={r.get_joint_pos()[1]:+.3f}")
    time.sleep(1)
