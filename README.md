# Master-s-Project

test_auto.py makes sure a drone is spawned, armed and takes off to 2.5m and lands back at home position on NVIDIA Isaac Sim using PX4 Autopilot Software


`~/isaacsim/python.sh test_auto.py`


`pip install sionna-rt pyzmq numpy`

`python3 sionna_rt_sensor_server.py`

`./python.sh -m pip install pyzmq`

`./python.sh -m pip install mavsdk`

`./python.sh isaac_px4_takeoff_land_with_sionna.py`



`sudo -E ./nr-softmodem --rfsim -O ~/ieee_ants2024_oai_tutorial/ran/conf/gnb.sa.band78.106prb.rfsim.conf`



`sudo -E ./nr-uesoftmodem -r 106 --numerology 1 --band 78 -C 3619200000 --rfsim --ssb 516 -O ~/ieee_ants2024_oai_tutorial/ran/conf/ue.conf`

`ping -I oaitun_ue1 192.168.70.135 `

`docker exec -it oai-ext-dn ping 10.0.0.3`

cat >/tmp/mosquitto.conf <<'EOF'
listener 1883 0.0.0.0
allow_anonymous true
EOF
nohup mosquitto -c /tmp/mosquitto.conf -v > /tmp/mosquitto.log 2>&1 &


`mosquitto_sub -h 127.0.0.1 -p 1883 -t pegasus/drone1/state -v`

`~/isaacsim/python.sh 1_px4_single_vehicle.py `




python3 scene_edit.py \
  --mavlink udpin://0.0.0.0:14540 \
  --mqtt-host 192.168.70.135 \
  --state-topic pegasus/drone1/state \
  --cmd-topic pegasus/drone1/cmd \
  --rate 2.0 \
  --ue-iface oaitun_ue1 \
  --extdn-ip 192.168.70.135 \
  --gnb-lat 40.423700 \
  --gnb-lon -86.921200 \
  --gnb-alt-m 0.0



