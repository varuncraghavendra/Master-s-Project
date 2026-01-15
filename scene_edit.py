#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import re
import shlex
import signal
import subprocess
import time
from typing import Optional, Dict, Any

from mavsdk import System
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion


# ---------------- MQTT ----------------

class MqttState:
    def __init__(self):
        self.connected = asyncio.Event()
        self.last_error: Optional[str] = None


def make_mqtt_client(state: MqttState) -> mqtt.Client:
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            state.connected.set()
        else:
            state.last_error = f"MQTT rejected: {reason_code}"
            state.connected.set()

    client.on_connect = on_connect
    return client


async def connect_mqtt(host: str, port: int, keepalive: int, timeout_s: float) -> mqtt.Client:
    state = MqttState()
    client = make_mqtt_client(state)
    client.loop_start()
    client.connect_async(host, port, keepalive)

    try:
        await asyncio.wait_for(state.connected.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise RuntimeError("MQTT connect timed out")

    if state.last_error:
        raise RuntimeError(state.last_error)

    print("MQTT connected.")
    return client


# ---------------- MAVSDK helpers ----------------

async def wait_connected(drone: System, timeout_s: float) -> bool:
    start = time.time()
    async for state in drone.core.connection_state():
        if state.is_connected:
            return True
        if time.time() - start >= timeout_s:
            return False


async def wait_global_position(drone: System, timeout_s: float) -> bool:
    start = time.time()
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            return True
        if time.time() - start >= timeout_s:
            return False


# ---------------- Linux / OAI tunnel helpers ----------------

def run_cmd(cmd: str, timeout: float = 2.0) -> str:
    parts = shlex.split(cmd)
    out = subprocess.check_output(parts, stderr=subprocess.STDOUT, timeout=timeout)
    return out.decode(errors="ignore")


def ping_rtt_ms(iface: str, dst: str) -> Optional[float]:
    try:
        out = run_cmd(f"ping -I {iface} -c 1 -W 1 {dst}", timeout=2.0)
        m = re.search(r"time=([0-9.]+)\s*ms", out)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None


def tunnel_stats(iface: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        out = run_cmd(f"ip -s link show {iface}", timeout=2.0)
        rx_m = re.search(
            r"RX:\s*bytes\s*packets\s*errors\s*dropped\s*overrun\s*mcast\s*\n\s*"
            r"([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)",
            out
        )
        tx_m = re.search(
            r"TX:\s*bytes\s*packets\s*errors\s*dropped\s*carrier\s*collsns\s*\n\s*"
            r"([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)",
            out
        )
        if rx_m:
            stats["rx"] = {
                "bytes": int(rx_m.group(1)),
                "packets": int(rx_m.group(2)),
                "errors": int(rx_m.group(3)),
                "dropped": int(rx_m.group(4)),
            }
        if tx_m:
            stats["tx"] = {
                "bytes": int(tx_m.group(1)),
                "packets": int(tx_m.group(2)),
                "errors": int(tx_m.group(3)),
                "dropped": int(tx_m.group(4)),
            }
    except Exception:
        pass
    return stats


# ---------------- RF metric estimation from distance ----------------
# We compute a "realistic" RF-style estimate that is *causally tied* to the drone position.

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0  # meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def fspl_db(distance_m: float, freq_hz: float) -> float:
    # Free-space path loss: FSPL(dB) = 20log10(d) + 20log10(f) + 32.44 (d in km, f in MHz)
    d_km = max(distance_m, 0.1) / 1000.0
    f_mhz = freq_hz / 1e6
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(f_mhz) + 32.44


def noise_floor_dbm(bw_hz: float, noise_figure_db: float) -> float:
    # Thermal noise: -174 dBm/Hz + 10log10(BW) + NF
    return -174.0 + 10.0 * math.log10(max(bw_hz, 1.0)) + noise_figure_db


def estimate_ue_metrics_from_distance(
    distance_m: float,
    freq_hz: float,
    tx_power_dbm: float,
    tx_gain_db: float,
    rx_gain_db: float,
    shadowing_db: float,
    bw_hz: float,
    noise_figure_db: float,
    interference_dbm: float,
    n_prb: int,
) -> Dict[str, Any]:
    pl = fspl_db(distance_m, freq_hz) + shadowing_db
    rsrp_dbm = tx_power_dbm + tx_gain_db + rx_gain_db - pl

    n0_dbm = noise_floor_dbm(bw_hz, noise_figure_db)
    # Simple SINR estimate with an interference term (can be set to very low if you want)
    # Combine noise+interference in linear mW:
    noise_mw = 10 ** (n0_dbm / 10.0)
    interf_mw = 10 ** (interference_dbm / 10.0)
    denom_mw = noise_mw + interf_mw
    sinr_db = rsrp_dbm - 10.0 * math.log10(max(denom_mw, 1e-12))

    # Very rough RSRQ estimate:
    # RSRQ = N * RSRP / RSSI  (linear); in dB: rsrq = rsrp + 10log10(N) - rssi
    # Approximate RSSI as received wideband power ~ rsrp + 10log10(N_PRB)
    # This is a simplified stand-in to make it distance-linked.
    rssi_dbm = rsrp_dbm + 10.0 * math.log10(max(n_prb, 1))
    rsrq_db = rsrp_dbm + 10.0 * math.log10(max(n_prb, 1)) - rssi_dbm  # cancels to ~0 with this approximation
    # To avoid trivial 0, we can include noise effect:
    rsrq_db = rsrq_db - max(0.0, (n0_dbm - rsrp_dbm) / 20.0)

    return {
        "distance_m": float(distance_m),
        "fspl_db": float(fspl_db(distance_m, freq_hz)),
        "pathloss_db": float(pl),
        "rsrp_dbm_est": float(rsrp_dbm),
        "sinr_db_est": float(sinr_db),
        "rsrq_db_est": float(rsrq_db),
        "noise_floor_dbm": float(n0_dbm),
        "bw_hz": float(bw_hz),
        "freq_hz": float(freq_hz),
    }


# ---------------- Telemetry state ----------------

last_pos = None
last_att = None
last_bat = None


async def read_position(drone: System):
    global last_pos
    async for p in drone.telemetry.position():
        last_pos = p


async def read_attitude(drone: System):
    global last_att
    async for a in drone.telemetry.attitude_euler():
        last_att = a


async def read_battery(drone: System):
    global last_bat
    async for b in drone.telemetry.battery():
        last_bat = b


# ---------------- Publish fused loop ----------------

async def publish_fused_loop(
    mqtt_client: mqtt.Client,
    state_topic: str,
    rate_hz: float,
    ue_iface: str,
    extdn_ip: str,
    gnb_lat: float,
    gnb_lon: float,
    gnb_alt_m: float,
    freq_hz: float,
    tx_power_dbm: float,
    tx_gain_db: float,
    rx_gain_db: float,
    shadowing_db: float,
    bw_hz: float,
    noise_figure_db: float,
    interference_dbm: float,
    n_prb: int,
):
    global last_pos, last_att, last_bat

    period = 1.0 / max(rate_hz, 0.1)
    count = 0

    while True:
        await asyncio.sleep(period)

        if last_pos is None:
            continue

        # Drone GPS from PX4
        d_lat = float(last_pos.latitude_deg)
        d_lon = float(last_pos.longitude_deg)
        d_alt = float(last_pos.absolute_altitude_m)

        horizontal = haversine_m(d_lat, d_lon, gnb_lat, gnb_lon)
        vertical = (d_alt - gnb_alt_m)
        dist3d = math.sqrt(horizontal * horizontal + vertical * vertical)

        ue_rf = estimate_ue_metrics_from_distance(
            distance_m=dist3d,
            freq_hz=freq_hz,
            tx_power_dbm=tx_power_dbm,
            tx_gain_db=tx_gain_db,
            rx_gain_db=rx_gain_db,
            shadowing_db=shadowing_db,
            bw_hz=bw_hz,
            noise_figure_db=noise_figure_db,
            interference_dbm=interference_dbm,
            n_prb=n_prb,
        )

        ue_net = {
            "iface": ue_iface,
            "extdn_ip": extdn_ip,
            "rtt_ms": ping_rtt_ms(ue_iface, extdn_ip),
            "tunnel": tunnel_stats(ue_iface),
        }

        drone_block = {
            "position_gps": {
                "lat_deg": d_lat,
                "lon_deg": d_lon,
                "abs_alt_m": d_alt,
                "rel_alt_m": float(last_pos.relative_altitude_m),
            },
            "attitude_euler_deg": None if last_att is None else {
                "roll": float(last_att.roll_deg),
                "pitch": float(last_att.pitch_deg),
                "yaw": float(last_att.yaw_deg),
            },
            "battery": None if last_bat is None else {
                "voltage_v": float(last_bat.voltage_v),
                "remaining_percent": float(last_bat.remaining_percent),
            },
        }

        payload = {
            "timestamp": time.time(),
            "drone": drone_block,
            "ue": {
                "network": ue_net,
                "rf_estimated_from_distance": ue_rf,
                "gnb_dummy_location": {
                    "lat_deg": gnb_lat,
                    "lon_deg": gnb_lon,
                    "alt_m": gnb_alt_m,
                },
            },
        }

        mqtt_client.publish(state_topic, json.dumps(payload), qos=0)

        count += 1
        if count % 5 == 0:
            print(f"Published fused state #{count} -> {state_topic}")


# ---------------- Command loop (joystick commands) ----------------

async def command_loop(drone: System, mqtt_client: mqtt.Client, cmd_topic: str):
    q: asyncio.Queue[str] = asyncio.Queue()

    def on_message(client, userdata, msg):
        try:
            s = msg.payload.decode(errors="ignore").strip()
            q.put_nowait(s)
        except Exception:
            pass

    mqtt_client.subscribe(cmd_topic, qos=0)
    mqtt_client.on_message = on_message
    print(f"Subscribed to commands: {cmd_topic}")

    while True:
        cmd = (await q.get()).strip()
        cmd_u = cmd.upper()

        try:
            if cmd_u == "ARM":
                print("CMD: ARM")
                await drone.action.arm()
            elif cmd_u == "DISARM":
                print("CMD: DISARM")
                await drone.action.disarm()
            elif cmd_u == "TAKEOFF":
                print("CMD: TAKEOFF")
                await drone.action.arm()
                await asyncio.sleep(0.2)
                await drone.action.takeoff()
            elif cmd_u == "LAND":
                print("CMD: LAND")
                await drone.action.land()
            else:
                print(f"CMD: Unknown: {cmd!r}")
        except Exception as e:
            print(f"Command failed ({cmd_u}): {e}")


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Pegasus/PX4 ↔ OAI via MQTT: fused (IsaacSim+UE metrics) + commands")

    # MAVSDK / PX4
    p.add_argument("--mavlink", default="udpin://0.0.0.0:14540")
    p.add_argument("--connect-timeout", type=float, default=20.0)
    p.add_argument("--position-timeout", type=float, default=20.0)

    # MQTT (broker on ext-dn)
    p.add_argument("--mqtt-host", default="192.168.70.135")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-keepalive", type=int, default=60)
    p.add_argument("--mqtt-connect-timeout", type=float, default=5.0)

    # Topics
    p.add_argument("--state-topic", default="pegasus/drone1/state")
    p.add_argument("--cmd-topic", default="pegasus/drone1/cmd")

    # Publish rate
    p.add_argument("--rate", type=float, default=2.0)

    # UE tunnel measurement
    p.add_argument("--ue-iface", default="oaitun_ue1")
    p.add_argument("--extdn-ip", default="192.168.70.135")

    # Dummy gNB location in GPS coords
    p.add_argument("--gnb-lat", type=float, required=True, help="Dummy gNB latitude (deg)")
    p.add_argument("--gnb-lon", type=float, required=True, help="Dummy gNB longitude (deg)")
    p.add_argument("--gnb-alt-m", type=float, default=0.0, help="Dummy gNB altitude (m)")

    # RF model knobs
    p.add_argument("--freq-hz", type=float, default=3.5e9)
    p.add_argument("--tx-power-dbm", type=float, default=46.0)     # ~40W macro-ish
    p.add_argument("--tx-gain-db", type=float, default=15.0)
    p.add_argument("--rx-gain-db", type=float, default=0.0)
    p.add_argument("--shadowing-db", type=float, default=6.0)
    p.add_argument("--bw-hz", type=float, default=20e6)
    p.add_argument("--noise-figure-db", type=float, default=7.0)
    p.add_argument("--interference-dbm", type=float, default=-120.0)
    p.add_argument("--n-prb", type=int, default=106)  # ~20MHz NR-ish PRB count

    return p.parse_args()


async def main():
    args = parse_args()

    drone = System()
    print(f"Connecting to MAVSDK at {args.mavlink} …")
    await drone.connect(system_address=args.mavlink)

    if not await wait_connected(drone, args.connect_timeout):
        raise RuntimeError("MAVSDK connection failed")
    print("MAVSDK connected.")

    if await wait_global_position(drone, args.position_timeout):
        print("Global position is valid; starting fused publish + command listener.")
    else:
        print("Warning: no global position yet; publishing anyway.")

    mqtt_client = await connect_mqtt(
        args.mqtt_host, args.mqtt_port, args.mqtt_keepalive, args.mqtt_connect_timeout
    )

    stop_event = asyncio.Event()

    def stop(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    tasks = [
        asyncio.create_task(read_position(drone)),
        asyncio.create_task(read_attitude(drone)),
        asyncio.create_task(read_battery(drone)),
        asyncio.create_task(
            publish_fused_loop(
                mqtt_client=mqtt_client,
                state_topic=args.state_topic,
                rate_hz=args.rate,
                ue_iface=args.ue_iface,
                extdn_ip=args.extdn_ip,
                gnb_lat=args.gnb_lat,
                gnb_lon=args.gnb_lon,
                gnb_alt_m=args.gnb_alt_m,
                freq_hz=args.freq_hz,
                tx_power_dbm=args.tx_power_dbm,
                tx_gain_db=args.tx_gain_db,
                rx_gain_db=args.rx_gain_db,
                shadowing_db=args.shadowing_db,
                bw_hz=args.bw_hz,
                noise_figure_db=args.noise_figure_db,
                interference_dbm=args.interference_dbm,
                n_prb=args.n_prb,
            )
        ),
        asyncio.create_task(command_loop(drone, mqtt_client, args.cmd_topic)),
    ]

    await stop_event.wait()

    for t in tasks:
        t.cancel()

    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())

