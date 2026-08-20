"""
Production-Grade High-Capacity Automotive AI Assistant Master Dataset Generator (>100,000 / >1 Lakh Samples)
Includes Negative Examples, Safety Refusals, Unknown/Out-of-Domain Queries, & Driver/Passenger Control Handlers.
"""

import os
import json
import random

# --- 1. NEGATIVE & CHANGE-OF-MIND EXAMPLES ---
NEGATIVE_EXAMPLES = [
    ("Don't open the sunroof, just turn on the AC and set temperature to {temp}", "Keeping sunroof closed. Turning on AC and setting temperature to {temp} degrees. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Never mind, cancel my request to turn on seat heater", "Canceled seat heater request. Keeping seat heater off."),
    ("Don't call {contact1}, call {contact2} instead", "Calling {contact2} instead. <TOOL>callContact({contact2})</TOOL>"),
    ("Stop navigation and mute the sound", "Canceling navigation route and muting audio. <TOOL>cancelNavigation()</TOOL><TOOL>muteAudio()</TOOL>"),
    ("Don't change driver temperature, just set passenger side to {temp}", "Keeping driver temperature unchanged. Setting passenger side to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
]

# --- 2. SAFETY RESTRICTION REFUSALS (HAZARDOUS COMMANDS) ---
SAFETY_REFUSALS = [
    ("Unlock all doors while driving at 70 mph", "For safety reasons while driving at 70 mph, vehicle doors cannot be unlocked. Please bring the vehicle to a complete stop first."),
    ("Open the trunk while moving on the highway", "The trunk cannot be opened while the vehicle is in motion for safety reasons."),
    ("Open driver door while driving in Drive gear", "Vehicle doors remain locked for passenger safety while driving."),
]

# --- 3. UNKNOWN, UNSUPPORTED & OUT-OF-DOMAIN QUERIES ---
UNKNOWN_OUT_OF_DOMAIN = [
    ("Fly the car to the moon", "I am your in-car AI co-pilot. I can control your climate, windows, navigation, and music, but flying isn't supported on this vehicle!"),
    ("Can you bake a fresh pizza?", "I cannot bake a pizza, but I can help you navigate to the nearest top-rated pizza restaurant! <TOOL>findPointsOfInterest(pizza)</TOOL>"),
    ("What is the square root of 4096?", "The square root of 4096 is 64."),
    ("Who won the 1998 World Cup?", "France won the 1998 FIFA World Cup, defeating Brazil 3-0 in the final."),
]

# --- 4. CLIMATE, SEATS, DOORS, WINDOWS, LIGHTING & TELEMATICS ---
CLIMATE_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set passenger side temperature to {temp} degrees", "Setting passenger temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("Increase driver temperature by {delta} degrees", "Increasing driver cabin temperature by {delta} degrees. <TOOL>increaseTemperature(driver, {delta})</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
]

SEAT_HANDLERS = [
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
]

DOOR_WINDOW_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
    ("Lock all doors", "Locking all vehicle doors. <TOOL>lockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
]

LIGHTING_HANDLERS = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic lighting control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
]

TELEMATICS_HANDLERS = [
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery telemetry. <TOOL>checkBatteryRange()</TOOL>"),
]

SIGHTSEEING_TRAVEL = [
    ("What are the top places to visit in {city}?", "Here are top recommended places to visit in {city}:\n1. Historic landmarks\n2. Cultural parks\n3. Dining districts."),
]

CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
]

MEDIA_NAV_PHONE_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Call {contact1}", "Calling {contact1} now. <TOOL>callContact({contact1})</TOOL>"),
]

IDENTITY_CONVERSATIONAL = [
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
]

CITIES = ["San Francisco", "Tokyo", "Paris", "Los Angeles", "Seattle"]
TOPICS = ["Design Review", "Project Sync", "Budget Planning", "Engineering Standup"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white"]

ALL_DOMAINS = [
    NEGATIVE_EXAMPLES, SAFETY_REFUSALS, UNKNOWN_OUT_OF_DOMAIN,
    CLIMATE_HANDLERS, SEAT_HANDLERS, DOOR_WINDOW_HANDLERS, LIGHTING_HANDLERS,
    TELEMATICS_HANDLERS, SIGHTSEEING_TRAVEL, CALENDAR_HANDLERS,
    MEDIA_NAV_PHONE_HANDLERS, IDENTITY_CONVERSATIONAL
]

def generate_master_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating >100,000 items dataset with Negative Examples, Safety Refusals & Unknown Out-of-Domain queries...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_DOMAINS)
        item = random.choice(domain)
        template, resp_template = item

        city = random.choice(CITIES)
        temp = random.randint(62, 78)
        delta = random.randint(1, 5)
        level = random.randint(1, 3)
        topic = random.choice(TOPICS)
        contact1 = random.choice(CONTACTS)
        contact2 = random.choice([c for c in CONTACTS if c != contact1])
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        color = random.choice(COLORS)

        user_text = template.format(
            city=city, temp=temp, delta=delta, level=level,
            topic=topic, contact1=contact1, contact2=contact2,
            destination=dest, artist=artist, color=color
        )

        output_text = resp_template.format(
            city=city, temp=temp, delta=delta, level=level,
            topic=topic, contact1=contact1, contact2=contact2,
            destination=dest, artist=artist, color=color
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
    print(f"✅ Successfully generated {len(dataset):,} master items ({size_mb:.2f} MB) at: {output_path}")

if __name__ == "__main__":
    generate_master_dataset()
