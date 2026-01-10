# Master-s-Project

test_auto.py makes sure a drone is spawned, armed and takes off to 2.5m and lands back at home position on NVIDIA Isaac Sim using PX4 Autopilot Software


~/isaacsim/python.sh test_auto.py


pip install sionna-rt pyzmq numpy
python3 sionna_rt_sensor_server.py

./python.sh -m pip install pyzmq
./python.sh -m pip install mavsdk
./python.sh isaac_px4_takeoff_land_with_sionna.py
