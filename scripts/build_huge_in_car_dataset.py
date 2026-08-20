"""
Production-Grade High-Capacity Automotive AI Assistant Master Dataset Generator (>100,000 / >1 Lakh Samples)
Balanced Positive, Negative, Refusal, Relative Delta, Disambiguation, and Out-of-Domain Examples for ALL Domains.
"""

import os
import json
import random

# --- 1. CLIMATE CONTROL (Positive, Negative, Refusal, Relative, Disambiguation) ---
CLIMATE_BALANCED = [
    # Positive
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set passenger side temperature to {temp} degrees", "Setting passenger temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    # Relative / Delta
    ("Make it a little warmer", "Raising driver temperature by 2 degrees. <TOOL>adjustTemperature(zone=\"driver\", delta=+2)</TOOL>"),
    ("Make it cooler in here", "Lowering cabin temperature by 2 degrees. <TOOL>adjustTemperature(zone=\"all\", delta=-2)</TOOL>"),
    # Negative / Cancellation
    ("Don't turn on the AC, just set fan speed to level {level}", "Keeping AC off. Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Never mind, don't change the temperature", "Keeping current temperature setting unchanged."),
    # Disambiguation
    ("Set temperature to 72 degrees", "Setting cabin temperature to 72 degrees for driver and passenger. <TOOL>setTemperature(zone=\"all\", target=72)</TOOL>"),
]

# --- 2. WINDOWS & SUNROOF (Positive, Negative, Refusal, Relative) ---
WINDOWS_SUNROOF_BALANCED = [
    # Positive
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Open the sunroof halfway", "Opening sunroof halfway. <TOOL>openSunroof(50)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    # Relative / Delta
    ("Open driver window a little more", "Opening driver window by 15 percent. <TOOL>setWindow(driver, delta=+15)</TOOL>"),
    # Negative / Cancellation
    ("Don't open the sunroof, just open the driver window 20 percent", "Keeping sunroof closed. Opening driver window to 20%. <TOOL>setWindow(driver, 20)</TOOL>"),
    # Refusal / Child Lock
    ("Roll down rear right window", "Rear window controls are currently locked by the driver. <TOOL>refuseAction(reason=\"child_lock_active\")</TOOL>"),
]

# --- 3. SEAT HEATERS & COOLERS (Positive, Negative, Diarization) ---
SEATS_BALANCED = [
    # Positive
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    # Negative / Cancellation
    ("Don't turn on seat heater, just turn on seat ventilation level {level}", "Setting driver seat cooling to level {level}. <TOOL>setSeatCooler(driver, {level})</TOOL>"),
    ("Never mind, turn off seat heater", "Turning off driver seat heater. <TOOL>setSeatHeater(driver, 0)</TOOL>"),
    # Diarization
    ("Turn on my seat ventilation", "Turning on rear-left seat ventilation. <TOOL>setSeatCooling(zone=\"rear_left\", level=3)</TOOL>"),
]

# --- 4. DOORS, TRUNK & FRUNK (Positive, Negative, Safety Refusals) ---
DOORS_BALANCED = [
    # Positive
    ("Lock all doors", "Locking all vehicle doors securely. <TOOL>lockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Open the frunk front trunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
    # Safety Refusal
    ("Open the rear trunk while driving at 45 mph", "I cannot open the trunk while the vehicle is in motion for safety reasons. <TOOL>refuseAction(reason=\"vehicle_in_motion\")</TOOL>"),
    ("Unlock all doors while driving at 70 mph", "Vehicle doors cannot be unlocked while driving for passenger safety. <TOOL>refuseAction(reason=\"vehicle_in_motion\")</TOOL>"),
]

# --- 5. MEDIA & AUDIO (Positive, Negative, Relative, Multi-Entity) ---
MEDIA_BALANCED = [
    # Positive
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause playback", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    # Relative / Delta
    ("Turn it up", "Increasing audio volume. <TOOL>adjustVolume(delta=+10)</TOOL>"),
    ("Make it quieter", "Decreasing audio volume. <TOOL>adjustVolume(delta=-10)</TOOL>"),
    # Multi-Entity
    ("Play the latest episode of {podcast} on Spotify at {vol_pct} percent volume", "Playing the latest episode of {podcast} on Spotify at {vol_pct}% volume now. <TOOL>playPodcast(spotify, {podcast})</TOOL><TOOL>setVolume({vol_pct})</TOOL>"),
    # Negative / Cancellation
    ("Don't play Spotify, play FM radio {radio_fm}", "Tuning radio to {radio_fm} FM. <TOOL>setRadioFrequency({radio_fm})</TOOL>"),
]

# --- 6. NAVIGATION & EV CHARGING (Positive, Negative, Preconditioning, Disambiguation) ---
NAV_EV_BALANCED = [
    # Positive
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Find Level 3 fast EV charger", "Locating Level 3 DC fast chargers along your route. <TOOL>findPointsOfInterest(ev_charger_level3)</TOOL>"),
    # EV Preconditioning
    ("Navigate to the fast charger on Route 9", "Setting route to Fast Charger on Route 9 and preconditioning battery for optimal charging speed. <TOOL>navigateTo(charger_id=\"RT9_FAST\")</TOOL><TOOL>setBatteryPreconditioning(true)</TOOL>"),
    # Disambiguation
    ("Navigate to Starbucks", "Found two Starbucks locations nearby: one on 4th Street, and one on Main Street. Which one should I route to?"),
    # Negative / Cancellation
    ("Cancel navigation route", "Canceling current navigation route. <TOOL>cancelNavigation()</TOOL>"),
]

# --- 7. PHONE & MESSAGING (Positive, Negative, Confirmation Loops) ---
COMMUNICATION_BALANCED = [
    # Positive
    ("Call {contact_fname}", "Calling {contact_fname} now. <TOOL>callContact({contact_fname})</TOOL>"),
    ("Read my unread text messages", "Reading your unread text messages. <TOOL>readUnreadMessages()</TOOL>"),
    # Confirmation Loop
    ("Text {contact_fname} 'Running 15 minutes late due to traffic'", "Ready to send {contact_fname}: 'Running 15 minutes late due to traffic'. Should I send it? <TOOL>draftMessage(recipient=\"{contact_fname}\", text=\"Running 15 minutes late due to traffic\")</TOOL>"),
    ("Yes, send it", "Message sent. <TOOL>sendMessage(draft_id=\"latest\")</TOOL>"),
    # Disambiguation
    ("Call {contact_fname}", "You have {contact_fname} Smith and {contact_fname} Chen in your contacts. Which one would you like to call?"),
]

# --- 8. TELEMATICS & SENSORS (Positive, Diagnostics) ---
TELEMATICS_BALANCED = [
    ("What is my tire pressure?", "Reading TPMS sensors: Front Left 35 PSI, Front Right 35 PSI, Rear Left 35 PSI, Rear Right 35 PSI. <TOOL>checkTirePressure()</TOOL>"),
    ("What is my EV battery state of charge?", "Checking battery state of charge: 85% battery remaining. <TOOL>checkBatteryRange()</TOOL>"),
    ("What does the yellow check engine light mean?", "The check engine light indicates a powertrain or emissions diagnostic fault. Running diagnostic scan. <TOOL>checkEngineStatus()</TOOL>"),
]

# --- 9. OUT-OF-DOMAIN & GENERAL KNOWLEDGE ---
OUT_OF_DOMAIN_BALANCED = [
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
    ("Fly the car to the moon", "I am your in-car AI co-pilot. I can control your climate, windows, navigation, and music, but flying isn't supported on this vehicle!"),
    ("What is the square root of 4096?", "The square root of 4096 is 64."),
    ("What are the top places to visit in Tokyo?", "Here are top recommended places to visit in Tokyo: 1. Senso-ji Temple 2. Tokyo Tower 3. Shinjuku Gyoen National Garden."),
]

CONTACT_FNAMES = ["Michael", "David", "Sarah", "Alex", "Emily", "John"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk"]
PODCASTS = ["Huberman Lab", "The Joe Rogan Experience", "TED Radio Hour", "Daily Tech News"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown"]
RADIO_FMS = ["98.1", "101.5", "104.5", "105.3"]

ALL_BALANCED_DOMAINS = [
    CLIMATE_BALANCED, WINDOWS_SUNROOF_BALANCED, SEATS_BALANCED,
    DOORS_BALANCED, MEDIA_BALANCED, NAV_EV_BALANCED,
    COMMUNICATION_BALANCED, TELEMATICS_BALANCED, OUT_OF_DOMAIN_BALANCED
]

def generate_final_balanced_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating >100,000 items FINAL BALANCED master dataset (Positive, Negative, Refusal, Relative, Disambiguation & Out-of-Domain)...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_BALANCED_DOMAINS)
        item = random.choice(domain)
        template, resp_template = item

        temp = random.randint(62, 78)
        level = random.randint(1, 3)
        vol_pct = random.randint(20, 80)
        fname = random.choice(CONTACT_FNAMES)
        artist = random.choice(ARTISTS)
        podcast = random.choice(PODCASTS)
        dest = random.choice(DESTINATIONS)
        radio_fm = random.choice(RADIO_FMS)

        user_text = template.format(temp=temp, level=level, vol_pct=vol_pct, contact_fname=fname, artist=artist, podcast=podcast, destination=dest, radio_fm=radio_fm)
        output_text = resp_template.format(temp=temp, level=level, vol_pct=vol_pct, contact_fname=fname, artist=artist, podcast=podcast, destination=dest, radio_fm=radio_fm)

        entry = {
            "instruction": sys_instruction,
            "user": user_text,
            "output": output_text
        }
        dataset.append(entry)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully generated {len(dataset):,} FINAL BALANCED master items ({size_mb:.2f} MB) at: {output_path}")

if __name__ == "__main__":
    generate_final_balanced_dataset()
