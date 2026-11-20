# F.R.I.D.A.Y. — Voice & Clap-Controlled Laboratory Automation (Master ESP8266)

An advanced, real-life F.R.I.D.A.Y. voice automation system inspired by Marvel's MCU AI assistant (voiced by Kerry Condon), powered by **Google Gemini (`gemini-3.8-flash`)**, studio cloud neural TTS (**Irish + Hindi Female Neural Voices**), and **hands-free double-clap wake detection** to control a Master ESP8266 laboratory automation controller running at `http://192.168.4.1`.

---

## 🌐 Multilingual Voice & NLP Support (English, Hindi & Hinglish)

F.R.I.D.A.Y. understands and speaks in **English**, **Hinglish** (Romanized Hindi), and **Hindi** (Devanagari):

* **Dual-Language Speech Recognition (STT)**: Recognizes Indian English, Romanized Hinglish, and pure Hindi Devanagari via dual-fallback recognition (`en-IN` + `hi-IN`).
* **Multilingual Neural Voices**:
  * **English Output**: `en-IE-EmilyNeural` (Authentic Irish female voice matching Kerry Condon).
  * **Hindi & Hinglish Output**: `hi-IN-SwaraNeural` (Natural, fluent Indian female voice).
* **Sample Voice Commands**:
  * **Lights**: *"Light on kar do"*, *"Lights band karo"*, *"लाइट चालू करो"*, *"Turn on the lights"*
  * **Fans**: *"Pankha chalu karo"*, *"Fans band kar do"*, *"पंखा चालू करो"*, *"Turn on the fans"*
  * **Master Controls**: *"Sab kuch on kar do"*, *"Pura lab band karo"*, *"सब कुछ चालू करो"*, *"Power up the lab"*
  * **Curtains**: *"Parde upar karo"*, *"Curtains niche karo"*, *"Parde roko"*, *"पर्दे ऊपर करो"*, *"Raise all curtains"*
  * **Individual Relays**: *"Fan 2 chalu karo"*, *"Light 3 on karo"*, *"Zone A band karo"*
  * **Status & Small Talk**: *"Status batao"*, *"Kya chal raha hai"*, *"Namaste Friday"*, *"Kaisi ho Friday"*

---

## ⚡ Master ESP Hardware & Lab Architecture

Controls all endpoints and zones defined in the master ESP web server (`index.html`):

1. **Master Controls**:
   - `ALL LIGHTS ON / OFF`: `GET http://192.168.4.1/master?type=lights&s=1|0`
   - `ALL FANS ON / OFF`: `GET http://192.168.4.1/master?type=fans&s=1|0`
   - `ALL FANS + LIGHTS ON / OFF`: `GET http://192.168.4.1/master?type=all&s=1|0`
2. **Zone Controls**:
   - `ZONE A [Relays 1-12]`: `GET http://192.168.4.1/zone?z=1&s=1|0`
   - `ZONE B [Fans 1-3, Lights 1-5]`: `GET http://192.168.4.1/zone?z=2&s=1|0`
   - `ZONE C [Relays 1-8]`: `GET http://192.168.4.1/zone?z=3&s=1|0`
3. **Individual Relays**:
   - `GET http://192.168.4.1/relay?z={1|2|3}&r={1..12}` (toggles specific relay channel)
4. **Motorized Curtains**:
   - Individual: `GET http://192.168.4.1/curtain?c={1..4}&a={1|2|0}` (1=Up, 2=Down, 0=Stop)
   - All Curtains: `GET http://192.168.4.1/allcurtains?a={1|2|0}`
5. **Real-Time HTML Status Parsing**:
   - Extracts live states of all 28 relays directly from the rendered HTML at `http://192.168.4.1/`.

---

## 🎙️ Real-Life F.R.I.D.A.Y. Persona & Speech

- **Authentic MCU Assistant Character**: Calm under pressure, razor-sharp, technically precise, and subtly witty.
- **Natural Delivery**: Does **NOT** repeat "Boss" in every sentence like a cartoon; speaks conversationally and directly, reserving "Boss" only for rare, contextual moments.
- **Zero Robotic Voice (`pyttsx3` prohibited)**: Studio cloud neural voices only (`hi-IN-SwaraNeural` and `en-IE-EmilyNeural`).
- **In-Memory Playback & Local Temp Caching**: Zero disk thrashing, sub-300ms audio response.

---

## 👏 Iron Man Style Hands-Free Clap Wake

- **No Startup Menus**: Starts silently on STANDBY immediately upon launching.
- **Double Clap**: Wakes F.R.I.D.A.Y. (*"All lab systems online. What do you need?"*) and enters hands-free voice mode automatically.
- **Standby Triggers**: Saying *"bye"*, *"goodbye"*, *"sleep"*, *"so jao"*, *"alvida"*, or *"standby"* speaks *"Standing by. Call me when you need me."* (or Hindi *"Theek hai, main standby par ja rahi hoon."*) and returns silently to standby.

---

## 🧠 Brain Architecture (`nlp_processor.py`)

- **Sub-Millisecond Fast Intent Matcher**: Direct hardware commands (lights, fans, zones, curtains, relays, and status checks in English, Hindi, and Hinglish) resolve in **< 0.5 ms** with zero cloud lag.
- **Primary Cloud Brain**: Gemini 3.8 Flash (`thinking_level="low"`) for open-ended queries with automatic multi-model failover (`gemini-3.8-flash` → `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → Local Backup).

---

## 🚀 Setup & Execution

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Configure Gemini API key in .env
Copy-Item .env.example .env
# Set GEMINI_API_KEY=<your_api_key> in .env

# 3. Run the integration test suite
.\.venv\Scripts\python.exe test_lab_integration.py

# 4. Launch F.R.I.D.A.Y.
.\.venv\Scripts\python.exe main.py
```

*Note: If the physical Master ESP is powered off, simply press **Enter** at the startup prompt to launch in offline **Simulation Mode** with full virtual lab hardware.*
