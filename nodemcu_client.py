"""
nodemcu_client.py
-----------------
Persistent HTTP client and controller for the Master ESP (ESP8266) smart lab
automation system located at `http://192.168.4.1`.

Implements all endpoints from `index.html`:
  1. Master Controls:
       GET /master?type={lights|fans|all}&s={1|0}
  2. Zone Controls:
       GET /zone?z={1|2|3}&s={1|0}
  3. Relay Controls:
       GET /relay?z={1|2|3}&r={1..12}
  4. Curtain Controls:
       GET /curtain?c={1..4}&a={1|2|0}   (1=Up, 2=Down, 0=Stop)
       GET /allcurtains?a={1|2|0}        (1=Up, 2=Down, 0=Stop)
  5. Live Status Extraction:
       Parses the rendered HTML from `http://<IP>/` to extract the real-time
       state of all 28 relays across Zones A, B, and C.

Includes offline Simulation Mode (press Enter if the physical ESP is off)
so all voice and NLP capabilities can be tested anywhere.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

import config

SIMULATION_BASE_URL = "SIMULATION"

# Persistent HTTP session with connection pooling
_HTTP_SESSION = requests.Session()

# In-memory virtual lab state used during Simulation Mode
_SIM_STATE: Dict[str, Any] = {
    "zone_1": {r: "ON" for r in range(1, 13)},
    "zone_2": {
        1: "OFF", 2: "OFF", 3: "OFF",  # Fans 1-3
        4: "ON", 5: "ON", 6: "ON", 7: "ON", 8: "ON",  # Lights 1-5
    },
    "zone_3": {r: "ON" for r in range(1, 9)},
    "curtains": {1: "STOPPED", 2: "STOPPED", 3: "STOPPED", 4: "STOPPED"},
}

_RELAY_HTML_PATTERN = re.compile(
    r"onclick='relay\((\d+),(\d+)\)'>([^<]+)<br><small>([^<]+)</small>",
    re.IGNORECASE,
)


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
    Check if a host is the Lab Master ESP by checking if `http://<host>/`
    returns the Lab Automation page or status endpoint.
    """
    base_url = _normalize_url(host)
    if not base_url or base_url == SIMULATION_BASE_URL:
        return None

    try:
        resp = requests.get(base_url, timeout=timeout)
        if resp.status_code == 200:
            text = resp.text
            if "Lab Automation" in text or "LAB AUTOMATION" in text or "relay(" in text:
                return base_url
    except (requests.RequestException, ValueError):
        pass

    # Secondary check on /status if implemented
    try:
        resp = requests.get(f"{base_url}/status", timeout=timeout)
        if resp.status_code == 200:
            return base_url
    except (requests.RequestException, ValueError):
        pass

    return None


def _get_local_wifi_ip() -> Optional[str]:
    """Detect the laptop's active IPv4 address."""
    for target in (("192.168.4.1", 80), ("8.8.8.8", 80), ("192.168.1.1", 80)):
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
    """Scan local /24 subnet for the Master ESP."""
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

    print(f"[Network]: Scanning {network} ({len(candidates)} hosts) for Lab Master ESP...")
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
    """Return clean host string."""
    if base_url == SIMULATION_BASE_URL:
        return SIMULATION_BASE_URL
    parsed = urlparse(_normalize_url(base_url))
    return parsed.hostname or base_url


def connect_to_nodemcu() -> str:
    """
    Connect to the Master ESP Lab Controller:
      1. Probe `http://192.168.4.1` (Master ESP default IP).
      2. Probe fallback IP / hostname from config.
      3. Scan local Wi-Fi subnet.
      4. Prompt user or press Enter for Simulation Mode.
    """
    primary_targets = [config.NODEMCU_HOST, config.NODEMCU_FALLBACK_IP, "192.168.4.1"]
    seen = set()

    for target in primary_targets:
        if not target or target in seen:
            continue
        seen.add(target)
        print(f"[Network]: Checking Master ESP at http://{target} ...")
        found = _probe_host(target, timeout=config.REQUEST_TIMEOUT)
        if found:
            print(f"[Network]: Master ESP connected at {found}")
            return found

    # Subnet scan
    local_ip = _get_local_wifi_ip()
    if local_ip:
        print(f"[Network]: Local Wi-Fi IP detected: {local_ip}")
        if not local_ip.startswith("192.168.4."):
            discovered = _scan_subnet(local_ip)
            if discovered:
                print(f"[Network]: Discovered Master ESP at {discovered}")
                return discovered

    # Manual or Simulation
    print("\n[Network]: Master ESP not detected at 192.168.4.1.")
    print("If connected to the Lab Wi-Fi, ensure your laptop is joined to the network.")
    print("Press Enter to run in Simulation Mode, or type the IP address:")

    try:
        typed = input("[Master ESP IP | Enter = Simulation Mode]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Network]: Running in offline Simulation Mode.")
        return SIMULATION_BASE_URL

    if not typed:
        print("[Network]: Running in offline Simulation Mode (virtual lab active).")
        return SIMULATION_BASE_URL

    verified = _probe_host(typed, timeout=config.REQUEST_TIMEOUT)
    if verified:
        print(f"[Network]: Verified Master ESP at {verified}")
        return verified

    candidate = _normalize_url(typed)
    print(f"[Warning]: {candidate} did not confirm 'Lab Automation'.")
    try:
        choice = input("Use this IP anyway? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return SIMULATION_BASE_URL

    return candidate if choice in {"y", "yes"} else SIMULATION_BASE_URL


# ---------------------------------------------------------------------------
# Lab Control Commands (Matching `index.html`)
# ---------------------------------------------------------------------------

def send_master(base_url: str, control_type: str, state: int) -> Tuple[bool, str]:
    """
    Master control command:
      GET /master?type={lights|fans|all}&s={1|0}
    """
    control_clean = control_type.strip().lower()
    if control_clean not in {"lights", "fans", "all"}:
        return False, f"Unknown master control type '{control_type}'."

    state_val = 1 if state else 0
    state_str = "ON" if state_val == 1 else "OFF"

    if base_url == SIMULATION_BASE_URL:
        if control_clean in {"lights", "all"}:
            for r in range(4, 9):
                _SIM_STATE["zone_2"][r] = state_str
        if control_clean in {"fans", "all"}:
            for r in range(1, 4):
                _SIM_STATE["zone_2"][r] = state_str
        if control_clean == "all":
            for r in range(1, 13):
                _SIM_STATE["zone_1"][r] = state_str
            for r in range(1, 9):
                _SIM_STATE["zone_3"][r] = state_str
        return True, f"Master command executed: all {control_clean} are now {state_str}."

    url = f"{_normalize_url(base_url)}/master?type={control_clean}&s={state_val}"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return True, f"Master: all {control_clean} set to {state_str}."
    except requests.RequestException as exc:
        return False, f"Master control failed: {exc}"


def send_zone(base_url: str, zone: int, state: int) -> Tuple[bool, str]:
    """
    Zone control command:
      GET /zone?z={1|2|3}&s={1|0}
    """
    if zone not in {1, 2, 3}:
        return False, f"Invalid zone {zone} (must be 1, 2, or 3)."

    state_val = 1 if state else 0
    state_str = "ON" if state_val == 1 else "OFF"
    zone_label = f"Zone {chr(64 + zone)}"

    if base_url == SIMULATION_BASE_URL:
        key = f"zone_{zone}"
        for r in _SIM_STATE[key]:
            _SIM_STATE[key][r] = state_str
        return True, f"{zone_label} is now completely {state_str}."

    url = f"{_normalize_url(base_url)}/zone?z={zone}&s={state_val}"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return True, f"{zone_label} set to {state_str}."
    except requests.RequestException as exc:
        return False, f"{zone_label} control failed: {exc}"


def send_relay(base_url: str, zone: int, relay: int) -> Tuple[bool, str]:
    """
    Individual relay toggle command:
      GET /relay?z={1|2|3}&r={1..12}
    """
    if zone not in {1, 2, 3}:
        return False, f"Invalid zone {zone}."

    max_relays = {1: 12, 2: 8, 3: 8}
    if relay < 1 or relay > max_relays.get(zone, 8):
        return False, f"Relay {relay} out of range for Zone {zone}."

    # Name lookup
    dev_name = config.LAB_ZONES.get(zone, {}).get("relays", {}).get(relay, f"Relay {relay}")

    if base_url == SIMULATION_BASE_URL:
        key = f"zone_{zone}"
        curr = _SIM_STATE[key].get(relay, "OFF")
        new_state = "OFF" if curr == "ON" else "ON"
        _SIM_STATE[key][relay] = new_state
        return True, f"{dev_name} (Zone {chr(64 + zone)}) is now {new_state}."

    url = f"{_normalize_url(base_url)}/relay?z={zone}&r={relay}"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return True, f"Toggled {dev_name} (Zone {chr(64 + zone)})."
    except requests.RequestException as exc:
        return False, f"Failed to switch {dev_name}: {exc}"


def send_curtain(base_url: str, curtain: int, action: int) -> Tuple[bool, str]:
    """
    Curtain control command:
      Individual: GET /curtain?c={1..4}&a={1|2|0}  (1=Up, 2=Down, 0=Stop)
      All:        GET /allcurtains?a={1|2|0}
    """
    action_labels = {1: "UP", 2: "DOWN", 0: "STOPPED"}
    action_str = action_labels.get(action, "STOPPED")

    if curtain == 0 or curtain == -1:  # 0 indicates all curtains
        if base_url == SIMULATION_BASE_URL:
            for c in _SIM_STATE["curtains"]:
                _SIM_STATE["curtains"][c] = action_str
            return True, f"All curtains set to {action_str}."

        url = f"{_normalize_url(base_url)}/allcurtains?a={action}"
        try:
            resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            return True, f"All curtains {action_str}."
        except requests.RequestException as exc:
            return False, f"All curtains command failed: {exc}"

    if curtain not in {1, 2, 3, 4}:
        return False, f"Curtain {curtain} invalid (must be 1-4)."

    curtain_name = config.LAB_CURTAINS.get(curtain, f"Curtain {curtain}")

    if base_url == SIMULATION_BASE_URL:
        _SIM_STATE["curtains"][curtain] = action_str
        return True, f"{curtain_name} set to {action_str}."

    url = f"{_normalize_url(base_url)}/curtain?c={curtain}&a={action}"
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return True, f"{curtain_name} {action_str}."
    except requests.RequestException as exc:
        return False, f"Curtain {curtain} failed: {exc}"


# ---------------------------------------------------------------------------
# Status Reading (HTML Parsing & Simulation)
# ---------------------------------------------------------------------------

def get_status(base_url: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Fetch the live status of all 28 relays and 4 curtains from the Master ESP.
    Parses the rendered HTML from `GET http://<base_url>/`.
    """
    if base_url == SIMULATION_BASE_URL:
        return True, dict(_SIM_STATE)

    url = _normalize_url(base_url)
    try:
        resp = _HTTP_SESSION.get(url, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    except requests.RequestException as exc:
        return False, {"error": str(exc)}

    # Parse relays from HTML buttons
    matches = _RELAY_HTML_PATTERN.findall(html)
    if not matches:
        return False, {"error": "Could not parse relay elements from page HTML."}

    parsed_status: Dict[str, Any] = {
        "zone_1": {},
        "zone_2": {},
        "zone_3": {},
        "curtains": {1: "READY", 2: "READY", 3: "READY", 4: "READY"},
    }

    for z_str, r_str, _, state_str in matches:
        try:
            z = int(z_str)
            r = int(r_str)
            zone_key = f"zone_{z}"
            if zone_key in parsed_status:
                parsed_status[zone_key][r] = state_str.strip().upper()
        except ValueError:
            continue

    return True, parsed_status
