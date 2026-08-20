"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Includes Sightseeing, Travel Recommendations, Detailed Technical Explanations, Local Food, & General Knowledge.
"""

import os
import json
import random

# --- 1. SIGHTSEEING, TRAVEL RECOMMENDATIONS & DETAILED EXPLANATIONS ---
SIGHTSEEING_TRAVEL = [
    (
        "What are the top places to visit in {city}?",
        "Here are top recommended places to visit in {city}:\n1. {attraction1}: Known for scenic views, historical significance, and vibrant atmosphere.\n2. {attraction2}: A famous landmark offering fantastic photo opportunities, dining, and cultural exhibits.\n3. {attraction3}: A beautiful park and relaxation spot perfect for a scenic walk or afternoon break.\nEnjoy your visit, and let me know if you would like navigation directions to any of these locations!"
    ),
    (
        "Can you recommend sightseeing spots around {city}?",
        "Certainly! Great sightseeing spots near {city} include {attraction1}, {attraction2}, and {attraction3}. Each offers unique sights, local history, and memorable experiences for travelers."
    ),
    (
        "Recommend some good local food and dining spots near {city}",
        "Near {city}, you'll find fantastic dining options! I recommend trying local favorites like {cuisine} at popular spots near {attraction1} or relaxed dining near {attraction2}."
    ),
]

TECHNICAL_EXPLANATIONS = [
    (
        "How does regenerative braking work in an electric vehicle?",
        "Regenerative braking in an EV converts kinetic energy back into electrical energy during deceleration. When you lift your foot off the accelerator or apply gentle braking, the electric motor reverses its operation to act as a generator, capturing energy that would otherwise be lost as heat and storing it back into the high-voltage battery to extend your driving range."
    ),
    (
        "Explain how adaptive cruise control works",
        "Adaptive Cruise Control (ACC) uses front-facing radar and camera sensors to automatically maintain a safe distance from the vehicle ahead. If traffic slows down, ACC automatically reduces your speed; when the road clears, it smoothly accelerates back to your set cruising speed."
    ),
    (
        "What should I do if my tire pressure warning light comes on while driving?",
        "If your TPMS tire pressure warning light comes on:\n1. Maintain firm control of the steering wheel and reduce your speed smoothly.\n2. Pull over safely to check all four tires for visible punctures or deflation.\n3. Use a tire pressure gauge or check the vehicle status display to verify PSI levels before driving to the nearest service station."
    ),
    (
        "How does lane keep assist help the driver?",
        "Lane Keep Assist uses windshield cameras to track road lane markings. If the system detects your vehicle drifting out of its lane without an active turn signal, it gently nudges the steering wheel or alerts you to keep your vehicle safely centered."
    ),
]

# --- 2. VEHICLE HARDWARE & CONTROL DOMAINS ---
LIGHTING_HANDLERS = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Turn off headlights", "Turning off headlights. <TOOL>turnOffHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic rain and light sensing headlight control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Turn on high beams", "Activating high beam headlights. <TOOL>turnOnHighBeams()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
]

SUNROOF_HANDLERS = [
    ("Open the sunroof", "Opening the sunroof for you. <TOOL>openSunroof(100)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Close the sunroof", "Closing the sunroof completely. <TOOL>closeSunroof()</TOOL>"),
]

SEAT_HEAT_COOL_HANDLERS = [
    ("Turn on driver seat heater level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Turn off driver seat heater", "Turning off driver seat heater. <TOOL>setSeatHeater(driver, 0)</TOOL>"),
]

DOOR_LOCK_HANDLERS = [
    ("Lock all doors", "Locking all vehicle doors securely. <TOOL>lockDoors()</TOOL>"),
    ("Unlock all doors", "Unlocking all vehicle doors. <TOOL>unlockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Open the frunk front trunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
]

WINDOW_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Roll down driver window", "Rolling down driver window completely. <TOOL>setWindow(driver, 100)</TOOL>"),
    ("Open windows slightly for fresh air", "Opening all windows slightly for fresh air. <TOOL>openWindowsSlightly(20)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
]

HVAC_DEFROST_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    ("Turn on rear window defroster", "Activating rear window defroster. <TOOL>turnOnRearDefroster()</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
]

CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
    ("Schedule a meeting titled {topic} with {contact} at {time_str} tomorrow", "Scheduling meeting {topic} with {contact} for {time_str} tomorrow. <TOOL>createEvent(title={topic}, contact={contact}, date=tomorrow, time={time_str})</TOOL>"),
    ("Join my online meeting now", "Opening your online meeting link now. <TOOL>joinMeeting()</TOOL>"),
]

MEDIA_NAV_EV_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Find nearest EV charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Start charging the vehicle", "Initiating EV charging session. <TOOL>startCharging()</TOOL>"),
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery state of charge. <TOOL>checkBatteryRange()</TOOL>"),
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
]

IDENTITY_CONVERSATIONAL = [
    ("Who are you?", "I am your in-car AI Assistant and co-pilot. I am here to help you with vehicle controls, navigation, music, calls, and travel recommendations."),
    ("What can you do in this car?", "I am your vehicle AI co-pilot! I can adjust climate, control windows and sunroof, play music, set navigation, check tire pressure and battery range, make calls, and execute vehicle diagnostics."),
]

# Random sampling dictionaries for sightseeing
CITIES = [
    ("San Francisco", "Golden Gate Bridge", "Fisherman's Wharf", "Golden Gate Park", "artisan seafood"),
    ("Tokyo", "Senso-ji Temple", "Tokyo Tower", "Shinjuku Gyoen National Garden", "authentic ramen and sushi"),
    ("Paris", "Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "fresh croissants and French bistro cuisine"),
    ("New York City", "Central Park", "Times Square", "Statue of Liberty", "New York style pizza"),
    ("London", "Big Ben", "Tower Bridge", "Hyde Park", "traditional fish and chips"),
    ("Kyoto", "Fushimi Inari Shrine", "Arashiyama Bamboo Grove", "Kinkaku-ji Golden Pavilion", "traditional Kaiseki dining"),
]

TOPICS = ["Design Review", "Project Sync", "Budget Planning", "Engineering Standup", "Quarterly Review"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David", "Sarah", "Alex"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Starbucks Coffee"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white"]
TIMES = ["9:00 AM", "10:30 AM", "2:00 PM", "4:15 PM", "5:00 PM"]

ALL_VEHICLE_DOMAINS = [
    SIGHTSEEING_TRAVEL, TECHNICAL_EXPLANATIONS, LIGHTING_HANDLERS, SUNROOF_HANDLERS,
    SEAT_HEAT_COOL_HANDLERS, DOOR_LOCK_HANDLERS, WINDOW_HANDLERS, HVAC_DEFROST_HANDLERS,
    CALENDAR_HANDLERS, MEDIA_NAV_EV_HANDLERS, IDENTITY_CONVERSATIONAL
]

def generate_complete_vehicle_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating 100,000+ items dataset covering Sightseeing, Detailed Q&A, Headlights, Sunroof, Seat Heaters, Doors, Windows, HVAC & Calendar...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_VEHICLE_DOMAINS)
        item = random.choice(domain)
        template, resp_template = item

        city_data = random.choice(CITIES)
        city, attr1, attr2, attr3, cuisine = city_data[0], city_data[1], city_data[2], city_data[3], city_data[4]

        temp = random.randint(62, 78)
        delta = random.randint(1, 5)
        level = random.randint(1, 3)
        topic = random.choice(TOPICS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        color = random.choice(COLORS)
        time_str = random.choice(TIMES)

        user_text = template.format(
            city=city, attraction1=attr1, attraction2=attr2, attraction3=attr3, cuisine=cuisine,
            temp=temp, delta=delta, level=level, topic=topic, contact=contact,
            destination=dest, artist=artist, color=color, time_str=time_str
        )

        output_text = resp_template.format(
            city=city, attraction1=attr1, attraction2=attr2, attraction3=attr3, cuisine=cuisine,
            temp=temp, delta=delta, level=level, topic=topic, contact=contact,
            destination=dest, artist=artist, color=color, time_str=time_str
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
    print(f"✅ Successfully generated {len(dataset):,} items ({size_mb:.2f} MB) covering Sightseeing, Detailed Q&A, Vehicle Controls & Calendar at: {output_path}")

if __name__ == "__main__":
    generate_complete_vehicle_dataset()
