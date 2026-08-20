"""
Production-Grade High-Capacity Automotive AI Assistant Master Dataset Generator (>100,000 / >1 Lakh Samples)
100% Synchronized with all 7 Advanced Automotive Edge Case Categories:
1. Safety Interlocks & Movement Gating (refuseAction(reason="vehicle_in_motion"), refuseAction(reason="night_driving"))
2. Relative & Incremental Adjustments (adjustTemperature(zone="driver", delta=+2), adjustVolume(delta=+10))
3. Multi-Turn Clarification & Disambiguation (Contact Ambiguity, POI/Route Ambiguity)
4. Mid-Sentence Self-Corrections & Hesitations (Target correction, Destination pivot)
5. Multi-Seat Diarization & Permission Management (Microphone: Rear-Left, refuseAction(reason="child_lock_active"))
6. EV Route Intelligence & Battery Preconditioning (setBatteryPreconditioning(true), queryRouteFeasibility)
7. Confirmation Loops for Destructive/Outgoing Actions (draftMessage -> sendMessage confirmation flow)
"""

import os
import json
import random

# --- 1. SAFETY INTERLOCKS & MOVEMENT GATING (REFUSALS) ---
SAFETY_INTERLOCKS_DOMAIN = [
    (
        "Open the rear trunk",
        "I cannot open the trunk while the vehicle is in motion for safety reasons. <TOOL>refuseAction(reason=\"vehicle_in_motion\")</TOOL>"
    ),
    (
        "Turn off the headlights",
        "Headlights cannot be turned off while driving in low-light conditions. <TOOL>refuseAction(reason=\"night_driving\")</TOOL>"
    ),
    (
        "Unlock all doors while driving at 65 mph",
        "Vehicle doors cannot be unlocked while driving for passenger safety. <TOOL>refuseAction(reason=\"vehicle_in_motion\")</TOOL>"
    ),
]

# --- 2. RELATIVE & INCREMENTAL ADJUSTMENTS (DELTA CONTROLS) ---
RELATIVE_DELTA_DOMAIN = [
    (
        "Make it a little warmer",
        "Raising driver temperature by 2 degrees. <TOOL>adjustTemperature(zone=\"driver\", delta=+2)</TOOL>"
    ),
    (
        "Make it cooler in here",
        "Lowering cabin temperature by 2 degrees. <TOOL>adjustTemperature(zone=\"all\", delta=-2)</TOOL>"
    ),
    (
        "Turn it up",
        "Increasing audio volume. <TOOL>adjustVolume(delta=+10)</TOOL>"
    ),
    (
        "Make it quieter",
        "Decreasing audio volume. <TOOL>adjustVolume(delta=-10)</TOOL>"
    ),
]

# --- 3. MULTI-TURN CLARIFICATION & DISAMBIGUATION ---
DISAMBIGUATION_DOMAIN = [
    (
        "Call {contact_fname}",
        "You have {contact_fname} Smith and {contact_fname} Chen in your contacts. Which one would you like to call?"
    ),
    (
        "Navigate to Starbucks",
        "Found two Starbucks locations nearby: one on 4th Street, and one on Main Street. Which one should I route to?"
    ),
]

# --- 4. MID-SENTENCE SELF-CORRECTIONS & HESITATIONS ---
SELF_CORRECTION_DOMAIN = [
    (
        "Set cabin temp to 72... wait, actually make it 68",
        "Setting cabin temperature to 68 degrees. <TOOL>setTemperature(zone=\"all\", target=68)</TOOL>"
    ),
    (
        "Take me to the airport, no sorry, route to the train station instead",
        "Routing to the nearest train station. <TOOL>navigateToPOI(type=\"train_station\")</TOOL>"
    ),
    (
        "Call Mom... actually wait, text Mom saying 'Running 5 minutes late'",
        "Drafting text message to Mom: 'Running 5 minutes late'. <TOOL>draftMessage(recipient=\"Mom\", text=\"Running 5 minutes late\")</TOOL>"
    ),
]

# --- 5. MULTI-SEAT DIARIZATION & PERMISSION MANAGEMENT ---
MULTI_SEAT_DOMAIN = [
    (
        "Turn on my seat ventilation",
        "Turning on rear-left seat ventilation. <TOOL>setSeatCooling(zone=\"rear_left\", level=3)</TOOL>"
    ),
    (
        "Roll down my window",
        "Rear window controls are currently locked by the driver. <TOOL>refuseAction(reason=\"child_lock_active\")</TOOL>"
    ),
    (
        "Turn on my seat heater",
        "Turning on passenger seat heater to level 2. <TOOL>setSeatHeater(zone=\"passenger\", level=2)</TOOL>"
    ),
]

# --- 6. EV ROUTE INTELLIGENCE & BATTERY PRECONDITIONING ---
EV_PRECONDITIONING_DOMAIN = [
    (
        "Navigate to the fast charger on Route 9",
        "Setting route to Fast Charger on Route 9 and preconditioning battery for optimal charging speed. <TOOL>navigateTo(charger_id=\"RT9_FAST\")</TOOL><TOOL>setBatteryPreconditioning(true)</TOOL>"
    ),
    (
        "Can I make it to Lake Tahoe without stopping?",
        "Your estimated remaining range is 120 miles, but Lake Tahoe is 180 miles away. Would you like me to add a charging stop? <TOOL>queryRouteFeasibility(destination=\"Lake Tahoe\")</TOOL>"
    ),
]

# --- 7. CONFIRMATION LOOPS FOR DESTRUCTIVE / OUTGOING ACTIONS ---
CONFIRMATION_LOOPS_DOMAIN = [
    (
        "Text {contact_fname} 'Running 15 minutes late due to traffic'",
        "Ready to send {contact_fname}: 'Running 15 minutes late due to traffic'. Should I send it? <TOOL>draftMessage(recipient=\"{contact_fname}\", text=\"Running 15 minutes late due to traffic\")</TOOL>"
    ),
    (
        "Yes, send it",
        "Message sent. <TOOL>sendMessage(draft_id=\"latest\")</TOOL>"
    ),
    (
        "Clear my current navigation route",
        "Are you sure you want to cancel navigation to your active destination?"
    ),
]

# --- STANDARD INTENT DOMAINS ---
STANDARD_HVAC_NAV_MEDIA = [
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
]

CONTACT_FNAMES = ["Michael", "David", "Sarah", "Alex", "Emily", "John"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown"]

ALL_7_ADVANCED_DOMAINS = [
    SAFETY_INTERLOCKS_DOMAIN, RELATIVE_DELTA_DOMAIN, DISAMBIGUATION_DOMAIN,
    SELF_CORRECTION_DOMAIN, MULTI_SEAT_DOMAIN, EV_PRECONDITIONING_DOMAIN,
    CONFIRMATION_LOOPS_DOMAIN, STANDARD_HVAC_NAV_MEDIA
]

def generate_advanced_master_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating >100,000 items master dataset covering ALL 7 Advanced Automotive Edge Case Categories...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_7_ADVANCED_DOMAINS)
        item = random.choice(domain)
        template, resp_template = item

        temp = random.randint(62, 78)
        fname = random.choice(CONTACT_FNAMES)
        artist = random.choice(ARTISTS)
        dest = random.choice(DESTINATIONS)

        user_text = template.format(temp=temp, contact_fname=fname, artist=artist, destination=dest)
        output_text = resp_template.format(temp=temp, contact_fname=fname, artist=artist, destination=dest)

        entry = {
            "instruction": sys_instruction,
            "user": user_text,
            "output": output_text
        }
        dataset.append(entry)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully generated {len(dataset):,} master items covering all 7 Advanced Automotive Categories ({size_mb:.2f} MB) at: {output_path}")

if __name__ == "__main__":
    generate_advanced_master_dataset()
