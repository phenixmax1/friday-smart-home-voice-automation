"""
test_lab_integration.py
-----------------------
Automated integration test for F.R.I.D.A.Y. Lab Voice Automation System:
- Verifies Master ESP client with Simulation Mode
- Verifies HTML parser extracting relay states from index.html
- Verifies NLP brain processing master, zones, relays, curtains, and small talk
- Verifies MCU F.R.I.D.A.Y. response quality (no repetitive 'Boss')
"""

import config
import nodemcu_client
import nlp_processor
import main

def test_simulation_controls():
    print("Testing Master ESP Simulation Mode...")
    base_url = nodemcu_client.SIMULATION_BASE_URL
    
    # 1. Initial status
    ok, status = nodemcu_client.get_status(base_url)
    assert ok, "Failed to get initial simulation status"
    assert "zone_1" in status and "zone_2" in status and "zone_3" in status and "curtains" in status
    print("  [Pass] Initial simulation status loaded successfully.")
    
    # 2. Master Controls
    ok, msg = nodemcu_client.send_master(base_url, "lights", 1)
    assert ok, f"send_master lights failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert status["zone_2"][4] == "ON", "Zone 2 Light 1 should be ON"
    print("  [Pass] Master lights ON verified.")
    
    ok, msg = nodemcu_client.send_master(base_url, "fans", 0)
    assert ok, f"send_master fans failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert status["zone_2"][1] == "OFF", "Zone 2 Fan 1 should be OFF"
    print("  [Pass] Master fans OFF verified.")
    
    ok, msg = nodemcu_client.send_master(base_url, "all", 1)
    assert ok, f"send_master all failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert status["zone_1"][1] == "ON" and status["zone_2"][1] == "ON"
    print("  [Pass] Master all ON verified.")
    
    # 3. Zone Controls
    ok, msg = nodemcu_client.send_zone(base_url, 1, 0)
    assert ok, f"send_zone 1 failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert all(status["zone_1"][r] == "OFF" for r in range(1, 13))
    print("  [Pass] Zone A full shutdown verified.")
    
    # 4. Individual Relay
    ok, msg = nodemcu_client.send_relay(base_url, 2, 2) # Fan 2 toggle
    assert ok, f"send_relay failed: {msg}"
    print("  [Pass] Individual relay toggle verified.")
    
    # 5. Curtain Controls
    ok, msg = nodemcu_client.send_curtain(base_url, 1, 1) # Curtain 1 UP
    assert ok, f"send_curtain failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert status["curtains"][1] == "UP"
    print("  [Pass] Curtain 1 UP verified.")
    
    ok, msg = nodemcu_client.send_curtain(base_url, 0, 0) # All curtains STOP
    assert ok, f"send_curtain all stop failed: {msg}"
    ok, status = nodemcu_client.get_status(base_url)
    assert all(status["curtains"][c] == "STOPPED" for c in range(1, 5))
    print("  [Pass] All curtains STOP verified.")


def test_nlp_commands():
    print("\nTesting F.R.I.D.A.Y. NLP & Persona...")
    dummy_status = {
        "zone_1": {r: "OFF" for r in range(1, 13)},
        "zone_2": {1: "OFF", 2: "OFF", 3: "OFF", 4: "OFF", 5: "OFF", 6: "OFF", 7: "OFF", 8: "OFF"},
        "zone_3": {r: "OFF" for r in range(1, 9)},
        "curtains": {1: "STOPPED", 2: "STOPPED", 3: "STOPPED", 4: "STOPPED"},
    }
    
    test_cases = [
        ("Turn on all the lights", "master", "lights", "on"),
        ("It's too hot in here", "master", "fans", "on"),
        ("Power up the lab", "master", "all", "on"),
        ("Shut down Zone B", "zone", "zone_2", "off"),
        ("Turn on Fan 2", "relay", "fan_2", "on"),
        ("Raise all curtains", "curtain", "all_curtains", "up"),
        ("Lower curtain 3", "curtain", "curtain_3", "down"),
        ("Stop curtains", "curtain", "all_curtains", "stop"),
        ("Status check", "status", "all", "status"),
    ]
    
    for phrase, exp_type, exp_target, exp_action in test_cases:
        res = nlp_processor._local_backup_brain(phrase, dummy_status)
        assert len(res.commands) > 0, f"No commands extracted for '{phrase}'"
        cmd = res.commands[0]
        assert cmd.target_type == exp_type, f"For '{phrase}': expected type {exp_type}, got {cmd.target_type}"
        assert cmd.action == exp_action, f"For '{phrase}': expected action {exp_action}, got {cmd.action}"
        print(f"  [Pass] '{phrase}' -> {cmd.target_type} {cmd.target} {cmd.action} | Reply: '{res.reply}'")

    # Small talk test
    res = nlp_processor._local_backup_brain("Hello Friday", dummy_status)
    assert len(res.commands) == 0
    assert "Boss" not in res.reply, "Spoken reply should avoid excessive 'Boss'"
    print(f"  [Pass] Small talk -> Reply: '{res.reply}'")


def test_hindi_hinglish_nlp():
    print("\nTesting Hindi & Hinglish NLP Support...")
    dummy_status = {
        "zone_1": {r: "OFF" for r in range(1, 13)},
        "zone_2": {1: "OFF", 2: "OFF", 3: "OFF", 4: "OFF", 5: "OFF", 6: "OFF", 7: "OFF", 8: "OFF"},
        "zone_3": {r: "OFF" for r in range(1, 9)},
        "curtains": {1: "STOPPED", 2: "STOPPED", 3: "STOPPED", 4: "STOPPED"},
    }

    hi_test_cases = [
        # Hinglish cases
        ("light on kar do", "master", "lights", "on"),
        ("lights band karo", "master", "lights", "off"),
        ("pankha chalu karo", "master", "fans", "on"),
        ("pankhe band kar do", "master", "fans", "off"),
        ("sab kuch on kar do", "master", "all", "on"),
        ("sab band kar do", "master", "all", "off"),
        ("parde upar karo", "curtain", "all_curtains", "up"),
        ("parde niche karo", "curtain", "all_curtains", "down"),
        ("curtains roko", "curtain", "all_curtains", "stop"),
        ("curtain 2 upar karo", "curtain", "curtain_2", "up"),
        ("zone a on karo", "zone", "zone_1", "on"),
        ("zone b band karo", "zone", "zone_2", "off"),
        ("fan 2 on karo", "relay", "fan_2", "on"),
        ("light 3 on karo", "relay", "light_3", "on"),
        ("status batao", "status", "all", "status"),
        # Pure Hindi Devanagari cases
        ("लाइट चालू करो", "master", "lights", "on"),
        ("पंखा बंद करो", "master", "fans", "off"),
        ("सब कुछ चालू करो", "master", "all", "on"),
        ("पर्दे ऊपर करो", "curtain", "all_curtains", "up"),
        ("स्टेटस बताओ", "status", "all", "status"),
    ]

    for phrase, exp_type, exp_target, exp_action in hi_test_cases:
        res = nlp_processor.understand_command(phrase, dummy_status)
        assert len(res.commands) > 0, f"No commands extracted for '{phrase}'"
        cmd = res.commands[0]
        assert cmd.target_type == exp_type, f"For '{phrase}': expected {exp_type}, got {cmd.target_type}"
        assert cmd.action == exp_action, f"For '{phrase}': expected action {exp_action}, got {cmd.action}"
        print(f"  [Pass] '{phrase}' -> {cmd.target_type} {cmd.target} {cmd.action} | Voice Reply: '{res.reply}'")

    # Small talk in Hindi / Hinglish
    for greeting in ["Namaste Friday", "Kaisi ho Friday", "नमस्ते", "tum kaun ho"]:
        res = nlp_processor.understand_command(greeting, dummy_status)
        assert len(res.commands) == 0
        assert len(res.reply) > 0
        print(f"  [Pass] Small talk '{greeting}' -> Voice Reply: '{res.reply}'")


def test_standby_phrase():
    print("\nTesting Standby & Sleep Triggers (English, Hindi & Hinglish)...")
    triggers = [
        "bye", "goodbye", "go to sleep", "sleep mode", "standby", "stop listening",
        "so jao", "sojao", "alvida", "chalo bye", "standby par jao", "अलविदा", "सो जाओ"
    ]
    for phrase in triggers:
        assert main.is_standby_phrase(phrase), f"Failed to detect standby phrase: '{phrase}'"
    print("  [Pass] All English, Hindi, and Hinglish standby triggers detected correctly.")


def test_html_parsing():
    print("\nTesting HTML parser against index.html...")
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    matches = nodemcu_client._RELAY_HTML_PATTERN.findall(html)
    assert len(matches) == 28, f"Expected 28 relays in index.html, found {len(matches)}"

    parsed = {1: {}, 2: {}, 3: {}}
    for z_str, r_str, _, state_str in matches:
        parsed[int(z_str)][int(r_str)] = state_str.strip().upper()

    assert len(parsed[1]) == 12, f"Zone 1 should have 12 relays, got {len(parsed[1])}"
    assert len(parsed[2]) == 8, f"Zone 2 should have 8 relays, got {len(parsed[2])}"
    assert len(parsed[3]) == 8, f"Zone 3 should have 8 relays, got {len(parsed[3])}"

    # Check Fan 1 state in Zone 2
    assert parsed[2][1] == "OFF", f"Expected Zone 2 Fan 1 to be OFF, got {parsed[2][1]}"
    # Check Light 1 state in Zone 2
    assert parsed[2][4] == "ON", f"Expected Zone 2 Light 1 to be ON, got {parsed[2][4]}"
    print("  [Pass] index.html successfully parsed: 28 relays categorized across Zones A, B, and C.")


if __name__ == "__main__":
    test_simulation_controls()
    test_nlp_commands()
    test_hindi_hinglish_nlp()
    test_standby_phrase()
    test_html_parsing()
    print("\nALL INTEGRATION & MULTILINGUAL TESTS PASSED SUCCESSFULLY!")
