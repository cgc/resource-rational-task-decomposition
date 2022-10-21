import os

def is_arm64_mac():
    name = os.uname()
    return name.sysname == 'Darwin' and name.machine == 'arm64'
