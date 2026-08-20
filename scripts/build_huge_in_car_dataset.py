"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Comprehensive coverage for Headlights, Sunroof, Seat Heaters, Open/Close Doors, Windows, Defrosters, Wipers, & Telematics.
"""

import os
import json
import random

# --- 1. HEADLIGHTS & LIGHTING ---
LIGHTING_HANDLERS = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Turn off headlights", "Turning off headlights. <TOOL>turnOffHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic rain and light sensing headlight control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Turn on high beams", "Activating high beam headlights. <TOOL>turnOnHighBeams()</TOOL>"),
    ("Turn off high beams", "Switching high beam headlights back to low beams. <TOOL>turnOffHighBeams()</TOOL>"),
    ("Turn on fog lights", "Activating front fog lights. <TOOL>turnOnFogLights()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
    ("Set ambient brightness to {brightness_pct} percent", "Setting ambient lighting brightness to {brightness_pct}%. <TOOL>setAmbientLightingBrightness({brightness_pct})</TOOL>"),
    ("Turn on cabin dome light", "Turning on overhead interior dome light. <TOOL>turnOnDomeLight()</TOOL>"),
    ("Turn off cabin dome light", "Turning off interior dome light. <TOOL>turnOffDomeLight()</TOOL>"),
]

# --- 2. SUNROOF & MOONROOF ---
SUNROOF_HANDLERS = [
    ("Open the sunroof", "Opening the sunroof for you. <TOOL>openSunroof(100)</TOOL>"),
    ("Open sunroof to {pct} percent", "Opening sunroof to {pct}%. <TOOL>openSunroof({pct})</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Close the sunroof", "Closing the sunroof completely. <TOOL>closeSunroof()</TOOL>"),
    ("Open the moonroof shade", "Opening moonroof sunshade. <TOOL>openSunroofShade()</TOOL>"),
    ("Close the moonroof shade", "Closing moonroof sunshade. <TOOL>closeSunroofShade()</TOOL>"),
]

# --- 3. SEAT HEATERS & SEAT COOLERS ---
SEAT_HEAT_COOL_HANDLERS = [
    ("Turn on driver seat heater level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat heater level {level}", "Setting passenger seat heater to level {level}. <TOOL>setSeatHeater(passenger, {level})</TOOL>"),
    ("Turn on rear left seat heater level {level}", "Setting rear left seat heater to level {level}. <TOOL>setSeatHeater(rear_left, {level})</TOOL>"),
    ("Turn off driver seat heater", "Turning off driver seat heater. <TOOL>setSeatHeater(driver, 0)</TOOL>"),
    ("Turn on driver seat cooler level {level}", "Setting driver seat cooling to level {level}. <TOOL>setSeatCooler(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Turn off all seat heaters", "Turning off all seat heaters. <TOOL>setSeatHeater(all, 0)</TOOL>"),
]

# --- 4. OPEN & CLOSE DOORS, TRUNK, FRUNK, LOCKS ---
DOOR_LOCK_HANDLERS = [
    ("Open driver door", "Unlocking and opening driver door. <TOOL>openDoor(driver)</TOOL>"),
    ("Open passenger door", "Unlocking and opening passenger door. <TOOL>openDoor(passenger)</TOOL>"),
    ("Close driver door", "Closing driver door. <TOOL>closeDoor(driver)</TOOL>"),
    ("Lock all doors", "Locking all vehicle doors securely. <TOOL>lockDoors()</TOOL>"),
    ("Unlock all doors", "Unlocking all vehicle doors. <TOOL>unlockDoors()</TOOL>"),
    ("Open the trunk", "Opening the rear trunk. <TOOL>openTrunk()</TOOL>"),
    ("Close the trunk", "Closing the rear trunk. <TOOL>closeTrunk()</TOOL>"),
    ("Open the frunk front trunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
    ("Close the frunk", "Closing the front trunk. <TOOL>closeFrunk()</TOOL>"),
    ("Enable rear child safety locks", "Activating rear door child safety locks. <TOOL>setChildLock(all, true)</TOOL>"),
]

# --- 5. OPEN & CLOSE WINDOWS ---
WINDOW_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Roll down driver window", "Rolling down driver window completely. <TOOL>setWindow(driver, 100)</TOOL>"),
    ("Roll up driver window", "Rolling up driver window completely. <TOOL>setWindow(driver, 0)</TOOL>"),
    ("Roll down passenger window to {pct} percent", "Opening passenger window to {pct}%. <TOOL>setWindow(passenger, {pct})</TOOL>"),
    ("Open windows slightly for fresh air", "Opening all windows slightly for fresh air. <TOOL>openWindowsSlightly(20)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
    ("Roll down all windows", "Rolling down all windows completely. <TOOL>setWindow(all, 100)</TOOL>"),
]

# --- 6. DEFROSTERS, WIPERS & CLIMATE ---
HVAC_DEFROST_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Increase temperature by {delta} degrees", "Increasing driver cabin temperature by {delta} degrees. <TOOL>increaseTemperature(driver, {delta})</TOOL>"),
    ("Decrease temperature by {delta} degrees", "Lowering cabin temperature by {delta} degrees. <TOOL>decreaseTemperature(driver, {delta})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    ("Turn on rear window defroster", "Activating rear window defroster. <TOOL>turnOnRearDefroster()</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Turn on heated steering wheel", "Activating heated steering wheel. <TOOL>setSteeringWheelHeater(true)</TOOL>"),
    ("Set windshield wipers to auto mode", "Setting windshield wipers to automatic rain sensing mode. <TOOL>setWindshieldWiperMode(auto)</TOOL>"),
    ("Fold side mirrors", "Folding exterior side mirrors. <TOOL>foldSideMirrors()</TOOL>"),
    ("Unfold side mirrors", "Unfolding exterior side mirrors. <TOOL>unfoldSideMirrors()</TOOL>"),
]

# --- 7. CALENDAR, MEETINGS & PRODUCTIVITY ---
CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("Do I have any meetings with {contact} today?", "Checking your calendar for meetings with {contact} today. <TOOL>searchEvents(topic={contact}, date=today)</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
    ("Am I free tomorrow morning around {time_str}?", "Checking your calendar availability for tomorrow morning at {time_str}. <TOOL>checkAvailability(date=tomorrow, time={time_str})</TOOL>"),
    ("Schedule a meeting titled {topic} with {contact} at {time_str} tomorrow", "Scheduling meeting {topic} with {contact} for {time_str} tomorrow. <TOOL>createEvent(title={topic}, contact={contact}, date=tomorrow, time={time_str})</TOOL>"),
    ("Cancel my meeting with {contact}", "Canceling your meeting with {contact}. <TOOL>deleteEvent(match={contact})</TOOL>"),
    ("Join my online meeting now", "Opening your online meeting link now. <TOOL>joinMeeting()</TOOL>"),
    ("Navigate to my next meeting location", "Starting navigation to your next meeting's physical location. <TOOL>navigateMeeting()</TOOL>"),
]

# --- 8. MEDIA, NAV, EV & TELEMATICS ---
MEDIA_NAV_EV_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Mute the sound", "Muting vehicle audio. <TOOL>muteAudio()</TOOL>"),
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Find nearest EV charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Start charging the vehicle", "Initiating EV charging session. <TOOL>startCharging()</TOOL>"),
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery state of charge. <TOOL>checkBatteryRange()</TOOL>"),
    ("Enable sentry security mode", "Enabling Sentry Mode vehicle security. <TOOL>enableSentryMode(true)</TOOL>"),
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
    ("Send text to {contact} saying {msg}", "Sending text message to {contact}. <TOOL>sendText({contact}, {msg})</TOOL>"),
]

# Random sampling dictionaries
TOPICS = ["Design Review", "Project Sync", "Budget Planning", "Engineering Standup", "Quarterly Review", "Client Call", "Product Roadmap"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David", "Sarah", "Alex", "Emily", "Dad"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Starbucks Coffee", "Target Superstore", "Yosemite National Park"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis", "Drake", "Ed Sheeran", "Billie Eilish"]
MESSAGES = ["I am driving home now", "Running 5 minutes late", "Heading to the office", "Pick up groceries on the way"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white", "amber"]
TIMES = ["9:00 AM", "10:30 AM", "2:00 PM", "4:15 PM", "5:00 PM", "6:30 PM"]

ALL_VEHICLE_DOMAINS = [
    LIGHTING_HANDLERS, SUNROOF_HANDLERS, SEAT_HEAT_COOL_HANDLERS, DOOR_LOCK_HANDLERS,
    WINDOW_HANDLERS, HVAC_DEFROST_HANDLERS, CALENDAR_HANDLERS, MEDIA_NAV_EV_HANDLERS
]

def generate_complete_vehicle_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating 100,000+ items dataset for Headlights, Sunroof, Seat Heaters, Doors, Windows, HVAC & Calendar...")
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

        temp = random.randint(62, 78)
        delta = random.randint(1, 5)
        level = random.randint(1, 3)
        pct = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        brightness_pct = random.randint(20, 100)
        topic = random.choice(TOPICS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        msg = random.choice(MESSAGES)
        color = random.choice(COLORS)
        time_str = random.choice(TIMES)

        user_text = template.format(
            temp=temp, delta=delta, level=level, pct=pct, brightness_pct=brightness_pct,
            topic=topic, contact=contact, destination=dest, artist=artist,
            msg=msg, color=color, time_str=time_str
        )

        output_text = resp_template.format(
            temp=temp, delta=delta, level=level, pct=pct, brightness_pct=brightness_pct,
            topic=topic, contact=contact, destination=dest, artist=artist,
            msg=msg, color=color, time_str=time_str
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
    print(f"✅ Successfully generated {len(dataset):,} items ({size_mb:.2f} MB) covering Headlights, Sunroof, Seat Heaters, Doors, Windows, HVAC & Calendar at: {output_path}")

if __name__ == "__main__":
    generate_complete_vehicle_dataset()
