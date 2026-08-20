"""
Production-Grade High-Capacity Automotive AI Assistant Master Dataset Generator (>100,000 / >1 Lakh Samples)
100% Synchronized with all 6 Structured Intent Domains:
1. CLIMATE_CONTROL (Implicit phrasing: "I'm shivering", "It's stuffy", AutoClimate, Defrost)
2. EV_CHARGING_AND_NAVIGATION (EV Constraints: "Level 3 only", "at least 150 kW", AddWaypoint, AvoidTolls)
3. MEDIA_AND_ENTERTAINMENT (Multi-entity requests: "Play latest episode of Huberman Lab on Spotify at 50% volume")
4. VEHICLE_HARDWARE_CONTROLS (Positional parameters: "rear left window", "driver side")
5. COMMUNICATION_AND_MESSAGING (Dictation with hesitation/correction: "Leaving now... wait, make that in 5 mins")
6. VEHICLE_STATUS_AND_DIAGNOSTICS (Natural conversational telemetry queries, Warning lights)
"""

import os
import json
import random

# --- 1. CLIMATE_CONTROL ---
CLIMATE_CONTROL_DOMAIN = [
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set passenger side temperature to {temp} degrees", "Setting passenger temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("I'm shivering, crank up the heat to {temp}", "I am turning up the heat to {temp} degrees to warm you up! <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("It's stuffy in here, turn on fresh air recirculation and fan speed to {level}", "Enabling fresh air recirculation and setting fan speed to {level}. <TOOL>setAirRecirculation(false)</TOOL><TOOL>setFanSpeed({level})</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Defrost windshield fast", "Activating maximum front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    ("Turn on driver seat heater to level {level}", "Setting driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooling level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Set climate control to automatic", "Enabling full automatic climate control. <TOOL>setAutoClimate(true)</TOOL>"),
]

# --- 2. EV_CHARGING_AND_NAVIGATION ---
EV_NAV_DOMAIN = [
    ("Find Level 3 fast EV charger with at least 150 kW and available plugs", "Locating Level 3 DC fast chargers with over 150 kW capacity and available plugs along your route. <TOOL>findPointsOfInterest(ev_charger_level3_150kw)</TOOL>"),
    ("Can I reach {destination} with my current battery range?", "Checking EV battery range: You currently have 85% battery with 236 miles range. {destination} is 180 miles away, so you will arrive with ~20% battery remaining!"),
    ("Navigate to {destination} avoiding tolls", "Starting navigation to {destination} avoiding toll roads. <TOOL>navigate({destination}, avoidTolls=true)</TOOL>"),
    ("Add waypoint for Starbucks along our route", "Adding Starbucks waypoint to your active route. <TOOL>addWaypoint(Starbucks)</TOOL>"),
    ("Check traffic ahead on our route", "Checking live traffic telemetry: Light traffic ahead with an estimated 3-minute delay. <TOOL>checkTraffic()</TOOL>"),
]

# --- 3. MEDIA_AND_ENTERTAINMENT ---
MEDIA_ENTERTAINMENT_DOMAIN = [
    ("Play the latest episode of {podcast} on Spotify at {vol_pct} percent volume", "Playing the latest episode of {podcast} on Spotify at {vol_pct}% volume now. <TOOL>playPodcast(spotify, {podcast})</TOOL><TOOL>setVolume({vol_pct})</TOOL>"),
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause playback", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Skip to next track", "Skipping to next track. <TOOL>skipTrack()</TOOL>"),
    ("Tune radio to {radio_fm} FM", "Tuning radio to {radio_fm} FM. <TOOL>setRadioFrequency({radio_fm})</TOOL>"),
    ("Switch audio source to Bluetooth", "Switching audio source to Bluetooth audio. <TOOL>setAudioSource(bluetooth)</TOOL>"),
]

# --- 4. VEHICLE_HARDWARE_CONTROLS ---
HARDWARE_CONTROLS_DOMAIN = [
    ("Roll down rear left window to 30 percent", "Opening rear left window to 30%. <TOOL>setWindow(rear_left, 30)</TOOL>"),
    ("Roll down driver side window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Open sunroof halfway", "Opening sunroof halfway. <TOOL>openSunroof(50)</TOOL>"),
    ("Set drive mode to Sport", "Engaging Sport drive mode. <TOOL>setDriveMode(sport)</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
    ("Open the rear trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Enable rear door child safety locks", "Activating rear child safety door locks. <TOOL>setChildLock(all, true)</TOOL>"),
]

# --- 5. COMMUNICATION_AND_MESSAGING ---
COMMUNICATION_DOMAIN = [
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
    ("Text {contact} saying 'Leaving now... wait, make that in five minutes'", "Sending text message to {contact}: 'Leaving now... make that in five minutes'. <TOOL>sendText({contact}, Leaving now... make that in 5 minutes)</TOOL>"),
    ("Read my unread text messages", "Reading your unread text messages. <TOOL>readUnreadMessages()</TOOL>"),
    ("Reply to {contact} saying 'On my way!'", "Replying to {contact}. <TOOL>sendText({contact}, On my way!)</TOOL>"),
    ("Decline incoming call", "Declining incoming call. <TOOL>declineCall()</TOOL>"),
]

# --- 6. VEHICLE_STATUS_AND_DIAGNOSTICS ---
STATUS_DIAGNOSTICS_DOMAIN = [
    ("What is my tire pressure?", "Reading TPMS sensors: Front Left 35 PSI, Front Right 35 PSI, Rear Left 35 PSI, Rear Right 35 PSI. <TOOL>checkTirePressure()</TOOL>"),
    ("What is my EV battery state of charge?", "Checking battery state of charge: 85% battery remaining. <TOOL>checkBatteryRange()</TOOL>"),
    ("What does the yellow check engine light mean?", "The check engine light indicates a powertrain or emissions diagnostic fault. Running diagnostic scan. <TOOL>checkEngineStatus()</TOOL>"),
    ("Schedule my next vehicle maintenance service", "Scheduling routine service appointment at your preferred dealership. <TOOL>scheduleService()</TOOL>"),
]

# --- 7. NEGATIVE EXAMPLES & SAFETY REFUSALS ---
NEGATIVE_SAFETY_DOMAIN = [
    ("Don't open the sunroof, just turn on the AC and set temperature to {temp}", "Keeping sunroof closed. Turning on AC and setting temperature to {temp} degrees. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Unlock all doors while driving at 70 mph", "For safety reasons while driving at 70 mph, vehicle doors cannot be unlocked. Please bring the vehicle to a complete stop first."),
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
]

PODCASTS = ["Huberman Lab", "The Joe Rogan Experience", "TED Radio Hour", "Daily Tech News", "NPR News", "Hardcore History"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis"]
CONTACTS = ["David", "Wife", "Mom", "John Smith", "Boss", "Sarah"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Yosemite National Park"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white"]
RADIO_FMS = ["98.1", "101.5", "104.5", "105.3"]

ALL_INTENT_DOMAINS = [
    CLIMATE_CONTROL_DOMAIN, EV_NAV_DOMAIN, MEDIA_ENTERTAINMENT_DOMAIN,
    HARDWARE_CONTROLS_DOMAIN, COMMUNICATION_DOMAIN, STATUS_DIAGNOSTICS_DOMAIN,
    NEGATIVE_SAFETY_DOMAIN
]

def generate_master_intent_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating >100,000 items master dataset covering ALL 6 Intent Domains & Phrasing Variations...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_INTENT_DOMAINS)
        item = random.choice(domain)
        template, resp_template = item

        temp = random.randint(62, 78)
        level = random.randint(1, 3)
        vol_pct = random.randint(20, 80)
        podcast = random.choice(PODCASTS)
        artist = random.choice(ARTISTS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        color = random.choice(COLORS)
        radio_fm = random.choice(RADIO_FMS)

        user_text = template.format(
            temp=temp, level=level, vol_pct=vol_pct, podcast=podcast,
            artist=artist, contact=contact, destination=dest, color=color, radio_fm=radio_fm
        )

        output_text = resp_template.format(
            temp=temp, level=level, vol_pct=vol_pct, podcast=podcast,
            artist=artist, contact=contact, destination=dest, color=color, radio_fm=radio_fm
        )

        entry = {
            "instruction": sys_instruction,
            "user": user_text,
            "output": output_text
        }
        dataset.append(entry)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully generated {len(dataset):,} master intent items ({size_mb:.2f} MB) at: {output_path}")

if __name__ == "__main__":
    generate_master_intent_dataset()
