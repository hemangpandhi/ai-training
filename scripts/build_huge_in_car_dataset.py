"""
Utility: Generator for Comprehensive Multimodal In-Car Vehicle Assistant Dataset
Expands in_car_dataset.json to cover 10 major automotive control domains with 3,000+ training pairs.
"""

import json
import random

CLIMATE_DOMAINS = [
    ("Turn on the air conditioning and set to {temp} degrees", "Setting the AC on and cabin temperature to {temp} degrees for you. <TOOL>turnOnAC()</TOOL><TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Set the temperature to {temp} in the car", "Adjusting cabin temperature to {temp} degrees. <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("I am feeling freezing, crank up the heat to {temp}", "I am turning up the heat to {temp} degrees for you! <TOOL>setTemperature(driver, {temp})</TOOL>"),
    ("Turn on driver seat heater to level {level}", "Turning on driver seat heater to level {level}. <TOOL>setSeatHeater(driver, {level})</TOOL>"),
    ("Turn on passenger seat cooler to level {level}", "Setting passenger seat cooling to level {level}. <TOOL>setSeatCooler(passenger, {level})</TOOL>"),
    ("Defrost the windshield and rear window", "Activating front and rear windshield defrosters. <TOOL>setDefroster(front)</TOOL><TOOL>setDefroster(rear)</TOOL>"),
    ("Set fan speed to level {level}", "Setting fan speed to level {level}. <TOOL>setFanSpeed({level})</TOOL>"),
]

WINDOW_DOMAINS = [
    ("Open the driver window halfway", "Opening driver window to 50%. <TOOL>setWindow(driver, 50)</TOOL>"),
    ("Roll down all windows completely", "Opening all windows completely. <TOOL>setWindow(all, 100)</TOOL>"),
    ("Close all windows", "Closing all windows securely. <TOOL>setWindow(all, 0)</TOOL>"),
    ("Open the sunroof halfway", "Opening sunroof halfway. <TOOL>setSunroof(50)</TOOL>"),
    ("Tilt the sunroof open", "Tilting sunroof open for ventilation. <TOOL>tiltSunroof()</TOOL>"),
    ("Lock all doors", "Locking all vehicle doors. <TOOL>lockDoors()</TOOL>"),
    ("Unlock the passenger doors", "Unlocking passenger doors. <TOOL>unlockDoors(passenger)</TOOL>"),
    ("Open the trunk", "Opening the trunk for you. <TOOL>openTrunk()</TOOL>"),
]

MEDIA_DOMAINS = [
    ("Play {artist} on Spotify", "Playing {artist} on Spotify now. <TOOL>playMusic(spotify, {artist})</TOOL>"),
    ("Pause the music", "Pausing audio playback. <TOOL>pauseMusic()</TOOL>"),
    ("Skip to the next song", "Skipping to the next track. <TOOL>skipTrack()</TOOL>"),
    ("Increase volume", "Increasing audio volume. <TOOL>increaseVolume()</TOOL>"),
    ("Mute the sound", "Muting vehicle audio. <TOOL>muteAudio()</TOOL>"),
    ("Play my driving playlist", "Playing your driving playlist. <TOOL>playMusic(playlist, driving)</TOOL>"),
]

NAVIGATION_DOMAINS = [
    ("Navigate to {destination}", "Setting navigation route to {destination}. <TOOL>setDestination({destination})</TOOL>"),
    ("Find nearest gas station", "Searching for nearby gas stations. <TOOL>findPointsOfInterest(gas_station)</TOOL>"),
    ("Find nearest EV charger", "Locating available EV fast chargers nearby. <TOOL>findPointsOfInterest(ev_charger)</TOOL>"),
    ("Cancel current navigation route", "Canceling current navigation route. <TOOL>cancelNavigation()</TOOL>"),
    ("Where is the nearest coffee shop?", "Finding coffee shops along your route. <TOOL>findPointsOfInterest(coffee)</TOOL>"),
]

TELEMATICS_DOMAINS = [
    ("What is my tire pressure?", "Checking tire pressure sensors. <TOOL>checkTirePressure()</TOOL>"),
    ("How much fuel do I have left?", "Reading fuel level telemetry. <TOOL>checkFuelLevel()</TOOL>"),
    ("What is my remaining battery range?", "Checking EV battery state of charge. <TOOL>checkBatteryRange()</TOOL>"),
    ("Is engine health okay?", "Running vehicle diagnostic check. <TOOL>checkEngineStatus()</TOOL>"),
]

DRIVING_MODES = [
    ("Switch drive mode to Sport", "Engaging Sport drive mode for enhanced throttle response. <TOOL>setDriveMode(sport)</TOOL>"),
    ("Switch drive mode to Eco", "Switching to Eco mode for maximum efficiency. <TOOL>setDriveMode(eco)</TOOL>"),
    ("Set regenerative braking to high", "Setting regenerative braking to maximum level. <TOOL>setRegenBraking(high)</TOOL>"),
    ("Set ambient lighting to blue", "Changing cabin ambient lighting to ocean blue. <TOOL>setAmbientLighting(blue)</TOOL>"),
]

DESTINATIONS = ["San Francisco", "Airport Terminal 2", "Home", "Office Downtown", "Starbucks", "Target Superstore", "Golden Gate Park"]
ARTISTS = ["Taylor Swift", "The Weeknd", "Coldplay", "Daft Punk", "Miles Davis", "Drake", "Ed Sheeran"]

def generate_huge_dataset(output_path="dataset/huge_in_car_dataset.json", total_samples=3000):
    dataset = []

    # System instruction template header
    sys_instruction = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

CRITICAL RULE: All tool tags MUST be placed sequentially at the VERY END of your response, after you finish speaking."""

    for i in range(total_samples):
        domain = random.choice([CLIMATE_DOMAINS, WINDOW_DOMAINS, MEDIA_DOMAINS, NAVIGATION_DOMAINS, TELEMATICS_DOMAINS, DRIVING_MODES])
        template, resp_template = random.choice(domain)

        temp = random.randint(62, 78)
        level = random.randint(1, 3)
        dest = random.choice(DESTINATIONS)
        artist = random.choice(ARTISTS)

        user_text = template.format(temp=temp, level=level, destination=dest, artist=artist)
        output_text = resp_template.format(temp=temp, level=level, destination=dest, artist=artist)

        entry = {
            "instruction": sys_instruction,
            "user": user_text,
            "output": output_text
        }
        dataset.append(entry)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated comprehensive huge dataset with {len(dataset)} items at: {output_path}")

if __name__ == "__main__":
    generate_huge_dataset()
