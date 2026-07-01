import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/veeti/dippa/ros2_ws/install/safe_shared_control'
