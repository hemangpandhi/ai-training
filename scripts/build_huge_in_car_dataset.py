"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Complete Integration of Climate/Dual-Zone/Defrosters, Headlights, Sunroof, Seat Heaters/Coolers, Doors, Windows,
Sightseeing, Travel, Calendar, ADAS, and Driver/Passenger Telematics.
"""

import os
import json
import random

# --- 1. CLIMATE, DUAL-ZONE & DEFROSTERS (VHAL 0x11600207, 291504647) ---
CLIMATE_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set passenger side temperature to {temp} degrees", "Setting passenger temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("I am feeling freezing, crank up the heat to {temp}", "I am turning up the heat to {temp} degrees for you! <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Increase driver temperature by {delta} degrees", "Increasing driver cabin temperature by {delta} degrees. <TOOL>increaseTemperature(driver, {delta})</TOOL>"),
    ("Decrease temperature by {delta} degrees", "Lowering cabin temperature by {delta} degrees. <TOOL>decreaseTemperature(driver, {delta})</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Turn on air recirculation", "Enabling cabin air recirculation. <TOOL>setAirRecirculation(true)</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    ("Turn on rear window defroster", "Activating rear window defroster. <TOOL>turnOnRearDefroster()</TOOL>"),
]

# --- 2. SEAT HEATERS & COOLERS (VHAL 356517131) ---
SEAT_HANDLERS = [
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat heater to level {level}", "Setting passenger seat heater to level {level}. <TOOL>setSeatHeater(passenger, {level})</TOOL>"),
    ("Turn on rear left seat heater to level {level}", "Setting rear left seat heater to level {level}. <TOOL>setSeatHeater(rear_left, {level})</TOOL>"),
    ("Turn on rear right seat heater to level {level}", "Setting rear right seat heater to level {level}. <TOOL>setSeatHeater(rear_right, {level})</TOOL>"),
    ("Turn on driver seat cooler to level {level}", "Setting driver seat cooling to level {level}. <TOOL>setSeatCooler(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Turn on heated steering wheel", "Activating heated steering wheel. <TOOL>setSteeringWheelHeater(true)</TOOL>"),
]

# --- 3. WINDOWS, SUNROOF, DOORS & LOCKS (VHAL 322964416, 371198722) ---
WINDOW_DOOR_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Roll down passenger window to {pct} percent", "Opening passenger window to {pct}%. <TOOL>setWindow(passenger, {pct})</TOOL>"),
    ("Open windows slightly for fresh air", "Opening all windows slightly for fresh air. <TOOL>openWindowsSlightly(20)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
    ("Open the sunroof halfway", "Opening sunroof halfway. <TOOL>openSunroof(50)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Close the sunroof", "Closing the sunroof completely. <TOOL>closeSunroof()</TOOL>"),
    ("Lock all doors", "Locking all vehicle doors. <TOOL>lockDoors()</TOOL>"),
    ("Unlock the doors", "Unlocking vehicle doors. <TOOL>unlockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Open the frunk front trunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
    ("Enable rear child safety locks", "Activating child safety door locks. <TOOL>setChildLock(all, true)</TOOL>"),
]

# --- 4. LIGHTING & HEADLIGHTS ---
LIGHTING_HANDLERS = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic lighting control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Turn on high beams", "Activating high beam headlights. <TOOL>turnOnHighBeams()</TOOL>"),
    ("Turn off high beams", "Switching high beam headlights back to low beams. <TOOL>turnOffHighBeams()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
]

# --- 5. TELEMATICS & SENSORS ---
TELEMATICS_HANDLERS = [
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("How much fuel do I have left?", "Reading fuel level telemetry. <TOOL>checkFuelLevel()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery telemetry. <TOOL>checkBatteryRange()</TOOL>"),
    ("Is engine health okay?", "Running vehicle diagnostic scan. <TOOL>checkEngineStatus()</TOOL>"),
]

# --- 6. SIGHTSEEING, TRAVEL & DETAILED EXPLANATIONS ---
SIGHTSEEING_TRAVEL = [
    (
        "What are the top places to visit in {city}?",
        "Here are top recommended places to visit in {city}:\n1. Famous landmarks and historic sites.\n2. Scenic cultural parks and photo spots.\n3. Vibrant local dining districts."
    ),
    (
        "How does regenerative braking work in an electric vehicle?",
        "Regenerative braking converts kinetic energy back into electrical energy during deceleration, capturing energy to charge the high-voltage battery."
    ),
]

# --- 7. CALENDAR & MEETINGS (CalendarToolHandler.kt) ---
CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
    ("Schedule a meeting titled {topic} with {contact} at {time_str} tomorrow", "Scheduling meeting {topic} with {contact} for {time_str} tomorrow. <TOOL>createEvent(title={topic}, contact={contact}, date=tomorrow, time={time_str})</TOOL>"),
    ("Join my online meeting now", "Opening your online meeting link now. <TOOL>joinMeeting()</TOOL>"),
]

# --- 8. MEDIA, NAV, PHONE & ADAS ---
MEDIA_NAV_PHONE_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Find nearest EV charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
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
TIMES = ["9:00 AM", "10:30 AM", "2:00 PM", "5:00 PM"]

ALL_DOMAINS = [
    CLIMATE_HANDLERS, SEAT_HANDLERS, WINDOW_DOOR_HANDLERS, LIGHTING_HANDLERS,
    TELEMATICS_HANDLERS, SIGHTSEEING_TRAVEL, CALENDAR_HANDLERS, MEDIA_NAV_PHONE_HANDLERS,
    IDENTITY_CONVERSATIONAL
]

def generate_full_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating 100,000+ items dataset with complete Climate Dual-Zone, Headlights, Sunroof, Seats, Doors & Windows...")
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
        pct = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        topic = random.choice(TOPICS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        color = random.choice(COLORS)
        time_str = random.choice(TIMES)

        user_text = template.format(
            city=city, temp=temp, delta=delta, level=level, pct=pct,
            topic=topic, contact=contact, destination=dest, artist=artist, color=color, time_str=time_str
        )

        output_text = resp_template.format(
            city=city, temp=temp, delta=delta, level=level, pct=pct,
            topic=topic, contact=contact, destination=dest, artist=artist, color=color, time_str=time_str
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
    print(f"✅ Successfully generated {len(dataset):,} items ({size_mb:.2f} MB) at: {output_path}")

if __name__ == "__main__":
    generate_full_dataset()
