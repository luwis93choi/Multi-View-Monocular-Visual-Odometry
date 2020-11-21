# Multi-View Monocular Visual Odometry
Monocular Visual Odometry Project using Conventional Multi-view Geometry with OpenCV and Python

-----------

# pyrealsense Jetson Nano Setup
>Reference <br>
>: Official Intel Realsense Python Wrapper (https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python) <br>
>: Fix by import (https://github.com/IntelRealSense/librealsense/issues/7540#issuecomment-709433690)

>Since Jetson Nano is ARM-based device, pyrealsense2 cannot be installed using pip. As a result, python wrapper of Intel Realsense SDK needs to be built from the source.

1. Ensure apt-get is up to date
- **sudo apt-get update && sudo apt-get upgrade**

2. Instally Python and its development files via apt-get (Python2 or Python3)
- For Python2 : **sudo apt-get install python python-dev**
- For Python3 : **sudo apt-get install python3 python3-dev**

3. Check python installation path
- For Python2 : **which python** (Copy the path)
- For Python3 : **which python3** (Copy the path)

4. Download librealsense & Move to librealsense directory
- **git clone https://github.com/IntelRealSense/librealsense.git**
- **cd librealsense**

5. Prepare build
- **mkdir build**
- **cd build**

6. Run top level CMake command with the following flags
- **cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DBUILD_EXECUTABLE=[full path of which python/which python3]**
> -DBUILD_PYTHON_BINDINGS : Decide whether to build python wrapper <br>
> -DBUILD_EXECUTABLE : Specify installation directory <br>

7. Make & Install
- **make -j1**
- **sudo make install**
> make -j1 : Use one core to build the library
> make -j4 : Use four core to build the library

8. At python script, append **/usr/local/lib/python3.6 (Python 3.6)** or **/usr/local/lib/python2.7 (Python 2.7)** in sys.path and import pyrealsense2.pyrealsense2 as rs

```python
import sys

sys.path.insert(0, '/usr/local/lib/python3.6')
print(sys.path)

import pyrealsense2.pyrealsense2 as rs

pipe = rs.pipeline()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, r$

pipeline.start(config)
```

> For an unknown reason, pyrealsense2 is installed at /usr/local/lib/python3.6 for python3 and /usr/local/lib/python2.7 for python2.This does not change even if -DBUILD_EXECUTABLE is changed. <br><br>
> As a result, it is recommend to append /usr/local/lib/python3.6 for python3 and /usr/local/lib/python2.7 for python2 as an easy solution. <br><br>
> Updating PYTHONPATH environment variable (export PYTHONPATH=$PYTHONPATH:/usr/local/lib) is an alternative method for importing pyrealsense2. However, in Jetson Nano, this does not seem to work well with this method. <br>
