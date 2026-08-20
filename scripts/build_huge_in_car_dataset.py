"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Includes Driver & Passenger Queries: Weather, Traffic, Rest Stops, Passenger Seat/Temp Controls, Dashboard Warnings,
EV Range Planning, Seat Massage, Kids/Trivia Road Trip Queries, & Sightseeing.
"""

import os
import json
import random

# --- 1. PASSENGER & REAR-SEAT SPECIFIC QUERIES ---
PASSENGER_QUERIES = [
    ("I am cold on my side, turn up passenger temperature to {temp} degrees", "Setting passenger cabin temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("Turn on passenger seat heater to level {level}", "Setting passenger seat heater to level {level}. <TOOL>setSeatHeater(passenger, {level})</TOOL>"),
    ("Turn on rear right seat heater level {level}", "Setting rear right seat heater to level {level}. <TOOL>setSeatHeater(rear_right, {level})</TOOL>"),
    ("Roll down passenger window 30 percent", "Opening passenger window to 30%. <TOOL>setWindow(passenger, 30)</TOOL>"),
    ("Turn on seat massage for driver", "Activating driver seat massage therapy mode. <TOOL>setSeatMassage(driver, true)</TOOL>"),
    ("Save my current seating position to Profile 1", "Saving current seat, mirror, and steering wheel positions to Memory Profile 1. <TOOL>saveSeatProfile(1)</TOOL>"),
]

# --- 2. ROAD TRIP, REST STOPS, WEATHER & TRAFFIC ---
ROAD_TRIP_QUERIES = [
    (
        "What's the weather going to be like when we arrive in {city}?",
        "The weather forecast for {city} currently shows clear skies with a comfortable temperature around 72°F. It will be great weather for your drive!"
    ),
    (
        "Is there heavy traffic ahead on our route?",
        "Checking live traffic telemetry: There is light traffic ahead with an estimated 3-minute delay. Your current ETA remains on schedule."
    ),
    ("Find clean rest stops and restrooms along our route", "Searching for top-rated rest stops and clean service plazas along your route. <TOOL>findPointsOfInterest(rest_stop)</TOOL>"),
    ("Find drive-thru coffee shops near our location", "Locating nearby drive-thru coffee shops. <TOOL>findPointsOfInterest(coffee_drivethru)</TOOL>"),
    ("How far is the next EV fast charger plaza?", "Checking EV charging telemetry: The nearest DC fast charger is 14 miles ahead off Exit 24. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
]

# --- 3. PASSENGER & KIDS ROAD TRIP ENTERTAINMENT & TRIVIA ---
ENTERTAINMENT_TRIVIA = [
    (
        "Tell us an interesting historical fact about {city}",
        "{city} has a rich history! Did you know that it became a major cultural and economic hub during the 19th century, attracting travelers and innovators from around the world?"
    ),
    ("Play kid-friendly music on Spotify", "Playing family and kid-friendly hits on Spotify! <TOOL>playMusic(spotify, Disney family hits)</TOOL>"),
    (
        "Tell us a road trip joke to pass the time",
        "Why did the computer take a car to work? Because it wanted to drive its hard drive!"
    ),
]

# --- 4. DASHBOARD WARNINGS, TIRE PSI & EV RANGE PLANNING ---
DASHBOARD_DIAGNOSTICS = [
    (
        "What does the yellow wrench warning light on my dashboard mean?",
        "The yellow wrench icon indicates a scheduled vehicle maintenance alert or secondary system check requirement. It is safe to drive, but you should schedule a routine service inspection soon."
    ),
    (
        "Can I make it to {destination} without stopping to charge?",
        "Checking EV battery telemetry: You currently have 85% charge with 236 miles of range. {destination} is approximately 180 miles away, so you will comfortably arrive with about 20% remaining battery capacity!"
    ),
    ("My windshield is fogging up rapidly, clear it fast!", "Activating maximum front windshield defroster and cabin dehumidifier now! <TOOL>turnOnFrontDefroster()</TOOL><TOOL>setFanSpeed(7)</TOOL>"),
    ("What is the recommended tire pressure for this vehicle?", "The recommended cold tire pressure for this vehicle is 35 PSI for all four tires. You can check your live PSI sensors anytime using vehicle telemetry.")
]

# --- 5. VEHICLE HARDWARE & CONTROL DOMAINS ---
LIGHTING_HANDLERS = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic lighting control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
]

SUNROOF_HANDLERS = [
    ("Open the sunroof", "Opening the sunroof for you. <TOOL>openSunroof(100)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Close the sunroof", "Closing the sunroof completely. <TOOL>closeSunroof()</TOOL>"),
]

DOOR_LOCK_HANDLERS = [
    ("Lock all doors", "Locking all vehicle doors securely. <TOOL>lockDoors()</TOOL>"),
    ("Unlock all doors", "Unlocking all vehicle doors. <TOOL>unlockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Open the frunk front trunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
]

WINDOW_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Open windows slightly for fresh air", "Opening all windows slightly for fresh air. <TOOL>openWindowsSlightly(20)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
]

HVAC_DEFROST_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
]

CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
    ("Join my online meeting now", "Opening your online meeting link now. <TOOL>joinMeeting()</TOOL>"),
]

MEDIA_NAV_EV_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
]

IDENTITY_CONVERSATIONAL = [
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
]

CITIES = ["San Francisco", "Tokyo", "Paris", "Los Angeles", "Seattle", "Chicago", "Boston", "Las Vegas"]
TOPICS = ["Design Review", "Project Sync", "Budget Planning", "Engineering Standup"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David", "Sarah"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Starbucks Coffee", "Yosemite National Park", "Los Angeles"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white"]

ALL_IN_CAR_QUERIES = [
    PASSENGER_QUERIES, ROAD_TRIP_QUERIES, ENTERTAINMENT_TRIVIA, DASHBOARD_DIAGNOSTICS,
    LIGHTING_HANDLERS, SUNROOF_HANDLERS, DOOR_LOCK_HANDLERS, WINDOW_HANDLERS,
    HVAC_DEFROST_HANDLERS, CALENDAR_HANDLERS, MEDIA_NAV_EV_HANDLERS, IDENTITY_CONVERSATIONAL
]

def generate_ultimate_in_car_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating >100,000 items dataset including Passenger Controls, Road Trips, Rest Stops, Weather, Dashboard Warnings & Battery Planning...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_IN_CAR_QUERIES)
        item = random.choice(domain)
        template, resp_template = item

        city = random.choice(CITIES)
        temp = random.randint(62, 78)
        delta = random.randint(1, 5)
        level = random.randint(1, 3)
        topic = random.choice(TOPICS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        color = random.choice(COLORS)

        user_text = template.format(
            city=city, temp=temp, delta=delta, level=level,
            topic=topic, contact=contact, destination=dest, artist=artist, color=color
        )

        output_text = resp_template.format(
            city=city, temp=temp, delta=delta, level=level,
            topic=topic, contact=contact, destination=dest, artist=artist, color=color
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
    print(f"✅ Successfully generated {len(dataset):,} items ({size_mb:.2f} MB) covering all Passenger & Driver In-Car Scenarios at: {output_path}")

if __name__ == "__main__":
    generate_ultimate_in_car_dataset()
