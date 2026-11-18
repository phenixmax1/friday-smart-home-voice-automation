"""
nodemcu_client.py
-----------------
Persistent HTTP client and concurrent /24 subnet auto-discovery for the
NodeMCU (ESP8266) smart home controller on the same local Wi-Fi network.

Also supports an offline Simulation Mode (press Enter at the manual IP prompt)
so you can test F.R.I.D.A.Y.'s voice and NLP brain even without the ESP8266
powered on.
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

import config

SIMULATION_BASE_URL = "SIMULATION"
VALID_STATES = frozenset({"on", "off", "toggle"})

# Persistent HTTP session for connection pooling and lower latency
_HTTP_SESSION = requests.Session()

# In-memory virtual relay state used only when running in Simulation Mode
_SIM_STATE: Dict[str, str] = {
    "id": config.EXPECTED_DEVICE_ID,
    "ip": "SIMULATION",
    "light": "OFF",
    "fan": "OFF",
    "relay3": "OFF",
    "relay4": "OFF",
}


def _normalize_url(host_or_url: str) -> str:
    """Format a host/IP into `http://<host>` with no trailing slash."""
    cleaned = host_or_url.strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.upper() == SIMULATION_BASE_URL:
        return SIMULATION_BASE_URL
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned


def _probe_host(host: str, timeout: float) -> Optional[str]:
    """
    Probe `GET http://<host>/status` and verify `"id": "nodemcu_smarthome"`.
    Returns the canonical base URL (preferring the NodeMCU's reported IP) or None.
    """
    base_url = _normalize_url(host)
    if not base_url or base_url == SIMULATION_BASE_URL:
        return None

    try:
        # Use standalone requests.get inside parallel scan threads (Session is not thread-safe)
        resp = requests.get(f"{base_url}/status", timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict) and data.get("id") == config.EXPECTED_DEVICE_ID:
            reported_ip = str(data.get("ip", "")).strip()
            return _normalize_url(reported_ip) if reported_ip else base_url
    except (requests.RequestException, ValueError):
        return None
    return None


def _get_local_wifi_ip() -> Optional[str]:
    """Detect the laptop's active Wi-Fi IPv4 address via the OS routing table."""
    for target in (("8.8.8.8", 80), ("192.168.1.1", 80), ("10.0.0.1", 80)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(target)
            local_ip = sock.getsockname()[0]
            if local_ip and not local_ip.startswith("127."):
                return local_ip
        except OSError:
            continue
        finally:
            sock.close()

    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                return ip
    except OSError:
        pass
    return None


def _scan_subnet(local_ip: str) -> Optional[str]:
    """
    Concurrently scan all 254 host IPs on the local /24 subnet using
    `ThreadPoolExecutor(max_workers=60)` with a 0.5s per-host timeout.
    """
    try:
        network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
    except ValueError:
        return None

    candidates: List[str] = [str(ip) for ip in network.hosts() if str(ip) != local_ip]
    found_event = threading.Event()

    def _worker(ip_addr: str) -> Optional[str]:
        if found_event.is_set():
            return None
        match = _probe_host(ip_addr, timeout=config.DISCOVERY_TIMEOUT)
        if match:
            found_event.set()
            return match
        return None

    print(f"[Network]: Scanning {network} ({len(candidates)} hosts) with {config.DISCOVERY_MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=config.DISCOVERY_MAX_WORKERS) as executor:
        futures = [executor.submit(_worker, ip) for ip in candidates]
        for future in as_completed(futures):
            try:
                discovered = future.result()
                if discovered:
                    return discovered
            except Exception:
                continue

    return None


def extract_host(base_url: str) -> str:
    """Return the clean IP/hostname from a base URL (or 'SIMULATION')."""
    if base_url == SIMULATION_BASE_URL:
        return SIMULATION_BASE_URL
    parsed = urlparse(_normalize_url(base_url))
    return parsed.hostname or base_url


def connect_to_nodemcu() -> str:
    """
    Connect to the NodeMCU on the shared Wi-Fi network:
      1. Check `http://{NODEMCU_HOST}/status` and `http://{NODEMCU_FALLBACK_IP}/status`.
      2. Scan the laptop's /24 Wi-Fi subnet concurrently (60 workers, 0.5s timeout).
      3. Prompt for the IP shown on the NodeMCU's 16x2 I2C LCD, or press Enter
         to launch in offline Simulation Mode.
    """
    # Step 1: Fast check of mDNS hostname and fallback IP
    for target in (config.NODEMCU_HOST, config.NODEMCU_FALLBACK_IP):
        if not target:
            continue
        print(f"[Network]: Probing http://{target}/status ...")
        found = _probe_host(target, timeout=config.REQUEST_TIMEOUT)
        if found:
            print(f"[Network]: NodeMCU connected at {found}")
            return found

    # Step 2: Concurrent /24 Wi-Fi subnet scan
    local_ip = _get_local_wifi_ip()
    if local_ip:
        print(f"[Network]: Local Wi-Fi IP detected: {local_ip}")
        discovered = _scan_subnet(local_ip)
        if discovered:
            print(f"[Network]: Auto-discovered NodeMCU at {discovered}")
            return discovered
    else:
        print("[Network]: Could not detect local Wi-Fi IPv4 address.")

    # Step 3: Manual I2C LCD IP entry (or Enter for Simulation Mode)
    print("\n[Network]: NodeMCU was not found automatically on this Wi-Fi network.")
    print("Type the IP shown on the NodeMCU's 16x2 I2C LCD, or press Enter for Simulation Mode.")

    while True:
        try:
            typed = input("[NodeMCU IP | Enter = Simulation Mode]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Network]: Starting in offline Simulation Mode.")
            return SIMULATION_BASE_URL

        if not typed:
            print("[Network]: Running in offline Simulation Mode (virtual relays active).")
            return SIMULATION_BASE_URL

        verified = _probe_host(typed, timeout=config.REQUEST_TIMEOUT)
        if verified:
            print(f"[Network]: Verified NodeMCU at {verified}")
            return verified

        candidate = _normalize_url(typed)
        print(f"[Warning]: {candidate} did not respond with id '{config.EXPECTED_DEVICE_ID}'.")
        try:
            choice = input("Use this IP anyway? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return SIMULATION_BASE_URL
        if choice in {"y", "yes"}:
            return candidate


def _simulate_command(device: str, state: str) -> Tuple[bool, str]:
    """Update virtual relay states when running in Simulation Mode."""
    targets = config.get_all_device_keys() if device == "all" else [device]
    for key in targets:
        if state == "toggle":
            _SIM_STATE[key] = "OFF" if _SIM_STATE.get(key) == "ON" else "ON"
        else:
            _SIM_STATE[key] = state.upper()

    if device == "all":
        return True, f"[SIM] All relays set to {state.upper()}."
    return True, f"[SIM] {config.get_device_name(device)} is now {_SIM_STATE[device]}."


def send_command(base_url: str, device: str, state: str) -> Tuple[bool, str]:
    """
    Send a relay control command:
        GET {base_url}/control?device={device}&state={state}&api=1
    """
    device_clean = device.strip().lower()
    state_clean = state.strip().lower()

    valid_devices = set(config.get_all_device_keys()) | {"all"}
    if device_clean not in valid_devices:
        return False, f"Unknown relay endpoint '{device_clean}'."
    if state_clean not in VALID_STATES:
        return False, f"Unsupported state '{state_clean}'."

    if base_url == SIMULATION_BASE_URL:
        return _simulate_command(device_clean, state_clean)

    url = f"{_normalize_url(base_url)}/control?device={device_clean}&state={state_clean}&api=1"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return False, f"Timed out contacting NodeMCU at {base_url}."
    except requests.exceptions.ConnectionError:
        return False, f"Lost connection to NodeMCU at {base_url}."
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        return False, f"NodeMCU returned HTTP error {code}."
    except requests.exceptions.RequestException as exc:
        return False, f"HTTP error: {exc}"

    try:
        payload = resp.json()
        if isinstance(payload, dict):
            if str(payload.get("status", "")).lower() == "error":
                return False, str(payload.get("message") or "NodeMCU rejected command.")
            msg = str(payload.get("message", "")).strip()
            if msg:
                return True, msg
    except ValueError:
        plain = resp.text.strip()
        if plain:
            return True, plain

    return True, f"{config.get_device_name(device_clean)} set to {state_clean.upper()}."


def get_status(base_url: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Fetch live states of all 4 relays from `GET {base_url}/status`.

    Returns `(True, status_dict)` on success or `(False, fallback_dict)` on failure.
    """
    if base_url == SIMULATION_BASE_URL:
        return True, dict(_SIM_STATE)

    url = f"{_normalize_url(base_url)}/status"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return True, data
    except (requests.RequestException, ValueError) as exc:
        return False, {
            "error": str(exc),
            "light": "UNKNOWN",
            "fan": "UNKNOWN",
            "relay3": "UNKNOWN",
            "relay4": "UNKNOWN",
        }

    return False, {
        "error": "Invalid JSON payload from NodeMCU.",
        "light": "UNKNOWN",
        "fan": "UNKNOWN",
        "relay3": "UNKNOWN",
        "relay4": "UNKNOWN",
    }
