# F.R.I.D.A.Y. — Voice & Clap-Controlled Smart Home Automation (NodeMCU ESP8266)

A modular Python voice assistant powered by **Google Gemini (`gemini-3.8-flash`)**, cloud female TTS (**Gemini 3.8 TTS + Edge Neural Irish F.R.I.D.A.Y. fallback**), and **double-clap wake detection** to control a 4-channel **NodeMCU (ESP8266)** smart home controller over shared Wi-Fi.

## Features

- **Double-Clap Wake & Standby**: Starts silently on standby; double-clap wakes F.R.I.D.A.Y. (*"All systems online."*) and activates hands-free voice mode. Say *"bye"* or *"quit"* to return silently to standby.
- **Fast Gemini AI Brain (`nlp_processor.py`)**: Uses the `google-genai` Interactions API with Pydantic Structured Outputs and automatic multi-model failover (`gemini-3.8-flash` → `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → local backup) to handle:
  - Direct & multi-device commands (*"Turn on the fan and switch off the light"*)
  - Contextual intents (*"It's too dark in here"*, *"It's really hot"*, *"Goodnight"*)
  - Negations (*"Don't turn on the fan"*)
  - Natural conversation and small talk
- **Cloud-Only Female Voice (`speech_engine.py`)**: Uses Google Gemini Cloud TTS (`gemini-3.8-flash-lite-tts`, voice `Kore`) with automatic fallback to Microsoft Edge Neural TTS (`en-IE-EmilyNeural`, Irish female F.R.I.D.A.Y. accent) — zero robotic on-device TTS.
- **Shared Wi-Fi Auto-Discovery (`nodemcu_client.py`)**: Automatically connects via `smarthome.local`, configured fallback IP, concurrent `/24` subnet scanning (`ThreadPoolExecutor`), manual I2C LCD IP entry, or offline Simulation Mode.
- **Dynamic Relay Mapping (`config.py`)**: Reassign physical appliances across `light`, `fan`, `relay3`, and `relay4` by editing `DEVICE_MAP`.

## NodeMCU HTTP API

- `GET http://<IP>/status`
  ```json
  {"id":"nodemcu_smarthome", "ip":"192.168.x.x", "light":"ON", "fan":"OFF", "relay3":"OFF", "relay4":"OFF"}
  ```
- `GET http://<IP>/control?device=<device>&state=<state>&api=1`
  - `device`: `light` | `fan` | `relay3` | `relay4` | `all`
  - `state`: `on` | `off` | `toggle`

## Setup & Run

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configure your Gemini API key
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY=<your_api_key>

# 3. Launch F.R.I.D.A.Y.
.\.venv\Scripts\python.exe main.py
```

## Project Structure

| File | Purpose |
| --- | --- |
| `config.py` | API settings, voice profiles, clap sensitivity, and `DEVICE_MAP` |
| `nodemcu_client.py` | Wi-Fi auto-discovery (`/24` scanner), persistent HTTP control, and simulation mode |
| `nlp_processor.py` | Gemini 3.8 structured NLP brain with multi-model failover and local backup |
| `speech_engine.py` | Cloud female TTS (Gemini TTS + Edge Neural fallback) and once-calibrated STT |
| `clap_detector.py` | Real-time double-clap detector with ambient calibration and echo rejection |
| `main.py` | Hands-free STANDBY ↔ ACTIVE voice controller loop |
| `requirements.txt` | Python dependencies |
