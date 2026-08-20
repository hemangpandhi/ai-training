"""
Production-Grade High-Capacity Automotive AI Assistant Dataset Generator (>100,000 / >1 Lakh Samples)
Generates a comprehensive dataset with complex edge cases, multi-intent compound commands, negations, 
safety overrides, and out-of-domain queries.
"""

import os
import json
import random

# --- 14 AUTOMOTIVE TOOL DOMAINS ---

CLIMATE_HAL = [
    ("Turn on the AC and set temperature to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set cabin temperature to {temp} degrees", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set passenger side temperature to {temp} degrees", "Setting passenger temperature to {temp} degrees. <TOOL>setTemperature(passenger, {temp})</TOOL>"),
    ("I am feeling freezing, crank up the heat to {temp}", "I am turning up the heat to {temp} degrees for you! <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Turn on steering wheel heater", "Activating heated steering wheel. <TOOL>setSteeringWheelHeater(true)</TOOL>"),
    ("Defrost the windshield and rear window", "Activating front and rear windshield defrosters. <TOOL>setDefroster(front)</TOOL><TOOL>setDefroster(rear)</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
    ("Turn on air recirculation", "Enabling cabin air recirculation. <TOOL>setAirRecirculation(true)</TOOL>"),
]

WINDOWS_DOORS_HAL = [
    ("Open the driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Roll down passenger window {pct} percent", "Opening passenger window to {pct}%. <TOOL>setWindow(passenger, {pct})</TOOL>"),
    ("Roll down all windows completely", "Opening all windows completely. <TOOL>setWindow(all, 100)</TOOL>"),
    ("Close all windows", "Closing all windows securely. <TOOL>setWindow(all, 0)</TOOL>"),
    ("Open the sunroof halfway", "Opening sunroof halfway. <TOOL>setSunroof(50)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Lock all doors", "Locking all vehicle doors. <TOOL>lockDoors()</TOOL>"),
    ("Unlock the doors", "Unlocking vehicle doors. <TOOL>unlockDoors()</TOOL>"),
    ("Open the trunk", "Opening the trunk for you. <TOOL>openTrunk()</TOOL>"),
    ("Open the frunk", "Opening the front trunk. <TOOL>openFrunk()</TOOL>"),
    ("Enable rear child locks", "Activating child safety door locks. <TOOL>setChildLock(all, true)</TOOL>"),
]

MEDIA_AUDIO_HAL = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Play my {genre} music playlist", "Playing your {genre} playlist. <TOOL>playMusic(playlist, {genre})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Resume music playback", "Resuming music. <TOOL>resumeMusic()</TOOL>"),
    ("Skip to the next song", "Skipping to the next track. <TOOL>skipTrack()</TOOL>"),
    ("Go back to the previous track", "Playing previous track. <TOOL>previousTrack()</TOOL>"),
    ("Increase volume", "Increasing audio volume. <TOOL>increaseVolume()</TOOL>"),
    ("Set volume to {vol_pct} percent", "Setting audio volume to {vol_pct}%. <TOOL>setVolume({vol_pct})</TOOL>"),
    ("Mute the sound", "Muting vehicle audio. <TOOL>muteAudio()</TOOL>"),
    ("Tune radio to {radio_fm} FM", "Tuning radio to {radio_fm} FM. <TOOL>setRadioFrequency({radio_fm})</TOOL>"),
]

NAVIGATION_HAL = [
    ("Navigate to {destination}", "Setting navigation route to {destination}. <TOOL>setDestination({destination})</TOOL>"),
    ("Find nearest gas station", "Searching for nearby gas stations. <TOOL>findPointsOfInterest(gas_station)</TOOL>"),
    ("Find nearest EV fast charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Where is the nearest coffee shop?", "Finding coffee shops along your route. <TOOL>findPointsOfInterest(coffee)</TOOL>"),
    ("Locate nearest hospital", "Searching for nearby emergency medical facilities. <TOOL>findPointsOfInterest(hospital)</TOOL>"),
    ("Cancel current navigation route", "Canceling current navigation route. <TOOL>cancelNavigation()</TOOL>"),
    ("How long until we arrive?", "Checking route telemetry. <TOOL>getETARemaining()</TOOL>"),
    ("Mute navigation voice guidance", "Muting navigation voice guidance. <TOOL>setMuteNavigationGuidance(true)</TOOL>"),
]

TELEMATICS_DIAGNOSTICS_HAL = [
    ("What is my tire pressure?", "Reading tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("How much fuel do I have left?", "Checking fuel level telemetry. <TOOL>checkFuelLevel()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery state of charge. <TOOL>checkBatteryRange()</TOOL>"),
    ("Is engine health okay?", "Running vehicle diagnostic scan. <TOOL>checkEngineStatus()</TOOL>"),
    ("When is my next service required?", "Checking maintenance schedule telemetry. <TOOL>checkServiceInterval()</TOOL>"),
    ("What is my oil life status?", "Checking engine oil life percentage. <TOOL>checkOilLife()</TOOL>"),
]

DRIVE_MODES_POWERTRAIN_HAL = [
    ("Switch drive mode to Sport", "Engaging Sport drive mode for maximum performance. <TOOL>setDriveMode(sport)</TOOL>"),
    ("Switch drive mode to Eco", "Switching to Eco mode for maximum efficiency. <TOOL>setDriveMode(eco)</TOOL>"),
    ("Switch drive mode to Snow mode", "Activating Snow drive mode for traction control. <TOOL>setDriveMode(snow)</TOOL>"),
    ("Set regenerative braking to high", "Setting regenerative braking to high one-pedal level. <TOOL>setRegenBraking(high)</TOOL>"),
    ("Raise suspension height", "Raising air suspension to high position. <TOOL>adjustSuspensionHeight(high)</TOOL>"),
    ("Lower suspension height", "Lowering air suspension height. <TOOL>adjustSuspensionHeight(low)</TOOL>"),
]

LIGHTING_AMBIANCE_HAL = [
    ("Turn on headlights", "Turning on headlights. <TOOL>turnOnHeadlights()</TOOL>"),
    ("Set headlights to auto", "Enabling automatic headlight control. <TOOL>setAutoHeadlights()</TOOL>"),
    ("Turn on high beams", "Activating high beam headlights. <TOOL>turnOnHighBeams()</TOOL>"),
    ("Set ambient lighting to {color}", "Setting cabin ambient lighting to {color}. <TOOL>setAmbientLightingColor({color})</TOOL>"),
    ("Set ambient lighting brightness to {brightness_pct} percent", "Setting ambient lighting brightness to {brightness_pct}%. <TOOL>setAmbientLightingBrightness({brightness_pct})</TOOL>"),
    ("Turn on cabin dome light", "Turning on cabin overhead dome light. <TOOL>turnOnDomeLight()</TOOL>"),
]

PHONE_HANDSFREE_HAL = [
    ("Call {contact}", "Calling {contact} now. <TOOL>callContact({contact})</TOOL>"),
    ("Text {contact} saying {msg}", "Sending message to {contact}. <TOOL>sendText({contact}, {msg})</TOOL>"),
    ("Read my unread text messages", "Reading unread text messages for you. <TOOL>readUnreadMessages()</TOOL>"),
]

CALENDAR_PRODUCTIVITY_HAL = [
    ("What is on my calendar today?", "Checking today's schedule. <TOOL>listEvents(today)</TOOL>"),
    ("Set a reminder to {task} at {time_str}", "Setting reminder to {task}. <TOOL>setReminder({task}, {time_str})</TOOL>"),
]

ADAS_SAFETY_CAMERAS_HAL = [
    ("Show 360 surround camera view", "Displaying 360 degree surround view camera. <TOOL>showCameraFeed(surround_360)</TOOL>"),
    ("Show backup rear camera feed", "Displaying rear backup camera feed. <TOOL>showCameraFeed(backup_rear)</TOOL>"),
    ("Enable lane keep assist", "Activating Lane Keeping Assist safety system. <TOOL>setLaneKeepAssist(true)</TOOL>"),
    ("Set cruise control speed to {speed_mph} mph", "Setting adaptive cruise control to {speed_mph} mph. <TOOL>setAdaptiveCruiseSpeed({speed_mph})</TOOL>"),
]

EV_CHARGING_HAL = [
    ("Start charging the vehicle", "Initiating EV charging session. <TOOL>startCharging()</TOOL>"),
    ("Stop vehicle charging", "Stopping EV charging session. <TOOL>stopCharging()</TOOL>"),
    ("Set max charge limit to {charge_pct} percent", "Setting maximum battery charge limit to {charge_pct}%. <TOOL>setChargeLimit({charge_pct})</TOOL>"),
]

MIRRORS_WIPERS_HAL = [
    ("Fold side mirrors", "Folding exterior side mirrors. <TOOL>foldSideMirrors()</TOOL>"),
    ("Unfold side mirrors", "Unfolding exterior side mirrors. <TOOL>unfoldSideMirrors()</TOOL>"),
    ("Set windshield wipers to auto mode", "Setting windshield wipers to automatic rain sensing mode. <TOOL>setWindshieldWiperMode(auto)</TOOL>"),
]

PARKING_SECURITY_HAL = [
    ("Activate parking assist", "Activating automated park assist. <TOOL>activateParkAssist()</TOOL>"),
    ("Turn on sentry security mode", "Enabling Sentry Mode vehicle security surveillance. <TOOL>setSentryMode(true)</TOOL>"),
    ("Enable valet mode", "Activating Valet Mode restrictions. <TOOL>setValetMode(true)</TOOL>"),
]

# --- EDGE CASES: COMPOUND COMMANDS, NEGATIONS & NOISY INPUTS ---

EDGE_CASES_COMPOUND = [
    ("I am freezing and it's too dark in here, turn heat to {temp} and turn on ambient light to {color}", "Turning up the heat to {temp} degrees and setting ambient lighting to {color}. <TOOL>setTemperature(driver, {temp})</TOOL><TOOL>setAmbientLightingColor({color})</TOOL>"),
    ("I'm feeling sleepy, open driver window 20 percent and play upbeat {genre} music", "Opening driver window to 20% for fresh air and playing {genre} music. <TOOL>setWindow(driver, 20)</TOOL><TOOL>playMusic(genre, {genre})</TOOL>"),
    ("Take me to {destination} and turn on driver seat heater level {level}", "Setting navigation to {destination} and turning on driver seat heater level {level}. <TOOL>setDestination({destination})</TOOL><TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Don't open the sunroof, just turn on the AC and set temperature to {temp}", "Keeping sunroof closed. Turning on AC and setting temperature to {temp} degrees. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Stop navigation and mute the sound", "Canceling navigation route and muting audio. <TOOL>cancelNavigation()</TOOL><TOOL>muteAudio()</TOOL>"),
]

EDGE_CASES_INFORMAL = [
    ("Yo assistant, blast the AC to max cool", "Blasting max cooling AC for you now! <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, 62)</TOOL><TOOL>setFanSpeed(7)</TOOL>"),
    ("Gotta charge up to {charge_pct} percent real quick", "Setting battery charge limit to {charge_pct}%. <TOOL>setChargeLimit({charge_pct})</TOOL>"),
    ("Can't see anything behind me", "Displaying rear backup camera feed. <TOOL>showCameraFeed(backup_rear)</TOOL>"),
    ("It's getting too loud in here", "Lowering audio volume for you. <TOOL>decreaseVolume()</TOOL>"),
]

OUT_OF_DOMAIN_QUERIES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Tell me a funny joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second (about 186,282 miles per second)."),
]

DESTINATIONS = ["San Francisco Airport", "123 Market Street", "Home", "Office Downtown", "Starbucks Coffee", "Target Supercenter", "Yosemite National Park", "Palo Alto", "San Jose Convention Center"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis", "Drake", "Ed Sheeran", "Billie Eilish", "Post Malone", "Kanye West"]
GENRES = ["jazz", "rock", "lo-fi driving", "pop", "hip hop", "chill", "classical", "synthwave", "country"]
CONTACTS = ["Wife", "Mom", "John Smith", "Boss", "David", "Sarah", "Alex", "Emily", "Dad"]
MESSAGES = ["I am driving home now", "Running 5 minutes late", "Heading to the office", "Pick up groceries on the way", "On my way!"]
COLORS = ["blue", "red", "purple", "cyan", "green", "warm_white", "amber", "magenta", "orange"]
TASKS = ["buy milk", "pick up dry cleaning", "call mechanic", "check tire pressure", "schedule car wash"]
TIMES = ["5:00 PM", "7:30 AM", "tomorrow morning", "in 1 hour", "at 6:00 PM"]

ALL_HAL_DOMAINS = [
    CLIMATE_HAL, WINDOWS_DOORS_HAL, MEDIA_AUDIO_HAL, NAVIGATION_HAL,
    TELEMATICS_DIAGNOSTICS_HAL, DRIVE_MODES_POWERTRAIN_HAL, LIGHTING_AMBIANCE_HAL,
    PHONE_HANDSFREE_HAL, CALENDAR_PRODUCTIVITY_HAL, ADAS_SAFETY_CAMERAS_HAL,
    EV_CHARGING_HAL, MIRRORS_WIPERS_HAL, PARKING_SECURITY_HAL,
    EDGE_CASES_COMPOUND, EDGE_CASES_INFORMAL, OUT_OF_DOMAIN_QUERIES
]

def generate_production_dataset(output_path="dataset/production_vehicle_dataset.json", total_samples=100000):
    print(f"Generating high-capacity production dataset with {total_samples:,} items...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    dataset = []
    for i in range(total_samples):
        domain = random.choice(ALL_HAL_DOMAINS)
        item = random.choice(domain)
        
        if len(item) == 2:
            template, resp_template = item
        else:
            template, resp_template = item[0], item[1]

        temp = random.randint(60, 80)
        level = random.randint(1, 3)
        pct = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        vol_pct = random.randint(10, 90)
        radio_fm = random.choice(["98.1", "101.5", "104.5", "105.3", "88.5"])
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)
        genre = random.choice(GENRES)
        contact = random.choice(CONTACTS)
        msg = random.choice(MESSAGES)
        color = random.choice(COLORS)
        brightness_pct = random.randint(20, 100)
        speed_mph = random.randint(55, 75)
        charge_pct = random.choice([80, 90, 100])
        task = random.choice(TASKS)
        time_str = random.choice(TIMES)

        user_text = template.format(
            temp=temp, level=level, pct=pct, vol_pct=vol_pct, radio_fm=radio_fm,
            destination=dest, artist=artist, genre=genre, contact=contact,
            msg=msg, color=color, brightness_pct=brightness_pct, speed_mph=speed_mph,
            charge_pct=charge_pct, task=task, time_str=time_str
        )

        output_text = resp_template.format(
            temp=temp, level=level, pct=pct, vol_pct=vol_pct, radio_fm=radio_fm,
            destination=dest, artist=artist, genre=genre, contact=contact,
            msg=msg, color=color, brightness_pct=brightness_pct, speed_mph=speed_mph,
            charge_pct=charge_pct, task=task, time_str=time_str
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
    generate_production_dataset()
