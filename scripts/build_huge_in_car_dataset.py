"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Strictly synchronized with the Android AIAssistant App Handlers (CalendarToolHandler, HVACToolHandler, MediaToolHandler, etc.).
"""

import os
import json
import random

# --- 1. CALENDAR & PRODUCTIVITY (CalendarToolHandler.kt) ---
CALENDAR_HANDLERS = [
    ("Search my calendar for {topic}", "Searching your calendar for {topic}. <TOOL>searchEvents(topic={topic})</TOOL>"),
    ("Do I have any meetings with {contact} today?", "Checking your calendar for meetings with {contact} today. <TOOL>searchEvents(topic={contact}, date=today)</TOOL>"),
    ("List all my upcoming meetings for today", "Here are your scheduled meetings for today. <TOOL>listEvents(date=today)</TOOL>"),
    ("Am I free tomorrow morning around {time_str}?", "Checking your calendar availability for tomorrow morning at {time_str}. <TOOL>checkAvailability(date=tomorrow, time={time_str})</TOOL>"),
    ("Get details for the meeting with {contact}", "Retrieving event details for your meeting with {contact}. <TOOL>getEventDetails(topic={contact})</TOOL>"),
    ("Schedule a meeting titled {topic} with {contact} at {time_str} tomorrow", "Scheduling meeting {topic} with {contact} for {time_str} tomorrow. <TOOL>createEvent(title={topic}, contact={contact}, date=tomorrow, time={time_str})</TOOL>"),
    ("Reschedule my meeting with {contact} to {time_str}", "Rescheduling your meeting with {contact} to {time_str}. <TOOL>updateEvent(match={contact}, newTime={time_str})</TOOL>"),
    ("Cancel my meeting with {contact}", "Canceling your meeting with {contact}. <TOOL>deleteEvent(match={contact})</TOOL>"),
    ("Cancel all my meetings for today", "Canceling all scheduled events for today. <TOOL>deleteAllEvents(date=today)</TOOL>"),
    ("Join my online meeting now", "Opening your online meeting link now. <TOOL>joinMeeting()</TOOL>"),
    ("Navigate to my next meeting location", "Starting navigation to your next meeting's physical location. <TOOL>navigateMeeting()</TOOL>"),
    ("Summarize the agenda for my meeting with {contact}", "Summarizing the meeting agenda notes for {contact}. <TOOL>summarizeMeeting(topic={contact})</TOOL>"),
    ("List upcoming birthdays this week", "Checking upcoming birthdays on your calendar. <TOOL>listBirthdays(horizon=7)</TOOL>"),
    ("Snooze my meeting reminder", "Snoozing your meeting notification. <TOOL>snoozeMeetingReminder()</TOOL>"),
]

# --- 2. COMMUNICATION & MESSAGING (CommunicationToolHandler.kt) ---
COMMUNICATION_HANDLERS = [
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
    ("Send text to {contact} saying {msg}", "Sending text message to {contact}. <TOOL>sendText({contact}, {msg})</TOOL>"),
    ("Read my unread text messages", "Reading your unread text messages. <TOOL>readUnreadMessages()</TOOL>"),
    ("Reply to {contact} saying {msg}", "Replying to {contact}. <TOOL>sendText({contact}, {msg})</TOOL>"),
]

# --- 3. CLIMATE & HVAC (HVACToolHandler.kt) ---
HVAC_HANDLERS = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Increase temperature by {delta} degrees", "Increasing driver cabin temperature by {delta} degrees. <TOOL>increaseTemperature(driver, {delta})</TOOL>"),
    ("Decrease temperature by {delta} degrees", "Lowering cabin temperature by {delta} degrees. <TOOL>decreaseTemperature(driver, {delta})</TOOL>"),
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Turn on front windshield defroster", "Activating front windshield defroster. <TOOL>turnOnFrontDefroster()</TOOL>"),
    ("Turn on rear window defroster", "Activating rear window defroster. <TOOL>turnOnRearDefroster()</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Turn on heated steering wheel", "Activating heated steering wheel. <TOOL>setSteeringWheelHeater(true)</TOOL>"),
]

# --- 4. WINDOWS & SUNROOF (WindowToolHandler.kt) ---
WINDOW_HANDLERS = [
    ("Open driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Open windows slightly for fresh air", "Opening all windows slightly for fresh air. <TOOL>openWindowsSlightly(20)</TOOL>"),
    ("Close all windows completely", "Closing all windows securely. <TOOL>closeWindowsCompletely()</TOOL>"),
    ("Open the sunroof halfway", "Opening sunroof halfway. <TOOL>openSunroof(50)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Close the sunroof", "Closing the sunroof. <TOOL>closeSunroof()</TOOL>"),
]

# --- 5. MEDIA & AUDIO (MediaToolHandler.kt) ---
MEDIA_HANDLERS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Play my {genre} playlist", "Playing your {genre} playlist. <TOOL>playMusic(playlist, {genre})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Resume music playback", "Resuming music. <TOOL>resumeMusic()</TOOL>"),
    ("Skip to the next song", "Skipping to the next track. <TOOL>skipTrack()</TOOL>"),
    ("Go back to the previous track", "Playing previous track. <TOOL>previousTrack()</TOOL>"),
    ("Increase volume", "Increasing audio volume. <TOOL>increaseVolume()</TOOL>"),
    ("Set volume to {vol_pct} percent", "Setting audio volume to {vol_pct}%. <TOOL>setVolume({vol_pct})</TOOL>"),
    ("Mute the sound", "Muting vehicle audio. <TOOL>muteAudio()</TOOL>"),
]

# --- 6. NAVIGATION & MAPS (NavigationToolHandler.kt) ---
NAV_HANDLERS = [
    ("Navigate to {destination}", "Starting navigation route to {destination}. <TOOL>navigate({destination})</TOOL>"),
    ("Find nearest gas station", "Searching for nearby gas stations. <TOOL>findPointsOfInterest(gas_station)</TOOL>"),
    ("Find nearest EV charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Cancel navigation route", "Canceling current navigation route. <TOOL>cancelNavigation()</TOOL>"),
    ("How long until we arrive?", "Checking route telemetry. <TOOL>getETARemaining()</TOOL>"),
]

# --- 7. EV & CHARGING (EVHandler.kt) ---
EV_HANDLERS = [
    ("Start charging the vehicle", "Initiating EV charging session. <TOOL>startCharging()</TOOL>"),
    ("Stop vehicle charging", "Stopping EV charging session. <TOOL>stopCharging()</TOOL>"),
    ("Set max charge limit to {charge_pct} percent", "Setting maximum battery charge limit to {charge_pct}%. <TOOL>setChargeLimit({charge_pct})</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery state of charge. <TOOL>checkBatteryRange()</TOOL>"),
]

# --- 8. SAFETY & CARE (SafetyAndCareHandler.kt) ---
SAFETY_HANDLERS = [
    ("Trigger emergency SOS help", "Triggering emergency SOS dispatch. <TOOL>triggerEmergencySOS()</TOOL>"),
    ("Enable sentry security mode", "Enabling Sentry Mode vehicle security. <TOOL>enableSentryMode(true)</TOOL>"),
    ("Activate valet mode", "Activating Valet Mode restrictions. <TOOL>activateValetMode(true)</TOOL>"),
]

# --- 9. SYSTEM TELEMATICS (SystemToolHandler.kt) ---
SYSTEM_HANDLERS = [
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("How much fuel do I have left?", "Checking fuel level telemetry. <TOOL>checkFuelLevel()</TOOL>"),
    ("Switch drive mode to Sport", "Engaging Sport drive mode. <TOOL>setDriveMode(sport)</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLighting({color})</TOOL>"),
]

# Random sampling dictionaries
TOPICS = ["Design Review", "Project Sync", "Budget Planning", "Engineering Standup", "Quarterly Review", "Client Call", "Product Roadmap"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David", "Sarah", "Alex", "Emily", "Dad"]
DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Starbucks Coffee", "Target Superstore", "Yosemite National Park"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis", "Drake", "Ed Sheeran", "Billie Eilish"]
GENRES = ["jazz", "rock", "lo-fi driving", "pop", "hip hop", "chill", "classical"]
MESSAGES = ["I am driving home now", "Running 5 minutes late", "Heading to the office", "Pick up groceries on the way"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white", "amber"]
TIMES = ["9:00 AM", "10:30 AM", "2:00 PM", "4:15 PM", "5:00 PM", "6:30 PM"]

ALL_APP_HANDLERS = [
    CALENDAR_HANDLERS, COMMUNICATION_HANDLERS, HVAC_HANDLERS, WINDOW_HANDLERS,
    MEDIA_HANDLERS, NAV_HANDLERS, EV_HANDLERS, SAFETY_HANDLERS, SYSTEM_HANDLERS
]

def generate_app_synchronized_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating 100,000+ items dataset strictly synchronized with Android AIAssistant App handlers...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_APP_HANDLERS)
        item = random.choice(domain)
        template, resp_template = item

        temp = random.randint(62, 78)
        delta = random.randint(1, 5)
        level = random.randint(1, 3)
        vol_pct = random.randint(10, 90)
        topic = random.choice(TOPICS)
        contact = random.choice(CONTACTS)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        genre = random.choice(GENRES)
        msg = random.choice(MESSAGES)
        color = random.choice(COLORS)
        time_str = random.choice(TIMES)
        charge_pct = random.choice([80, 90, 100])

        user_text = template.format(
            temp=temp, delta=delta, level=level, vol_pct=vol_pct,
            topic=topic, contact=contact, destination=dest, artist=artist,
            genre=genre, msg=msg, color=color, time_str=time_str, charge_pct=charge_pct
        )

        output_text = resp_template.format(
            temp=temp, delta=delta, level=level, vol_pct=vol_pct,
            topic=topic, contact=contact, destination=dest, artist=artist,
            genre=genre, msg=msg, color=color, time_str=time_str, charge_pct=charge_pct
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
    print(f"✅ Successfully generated {len(dataset):,} items ({size_mb:.2f} MB) synchronized with Android app at: {output_path}")

if __name__ == "__main__":
    generate_app_synchronized_dataset()
