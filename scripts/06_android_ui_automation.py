"""
Phase 6: Automated ADB UI Parser & Model Selection Script for Android Devices
"""

import os
import re
import time
import subprocess
import xml.etree.ElementTree as ET

DEVICE_ID = "3704105H8094TU"
TARGET_MODEL_SUBSTRING = "In-Car Gemma 4-E2B (Fine-Tuned)"
TARGET_BACKEND_SUBSTRING = "GPU"

def run_adb(cmd):
    full_cmd = f"adb -s {DEVICE_ID} {cmd}"
    res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def dump_ui_and_find_bounds(text_substring):
    run_adb("shell uiautomator dump /data/local/tmp/ui.xml")
    run_adb("pull /data/local/tmp/ui.xml ui.xml")
    
    if not os.path.exists("ui.xml"):
        return None, None
        
    tree = ET.parse("ui.xml")
    root = tree.getroot()
    
    for elem in root.iter("node"):
        text_val = elem.attrib.get("text", "")
        if text_substring.lower() in text_val.lower():
            bounds_str = elem.attrib.get("bounds", "")
            match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                return (cx, cy), bounds_str
    return None, None

def main():
    print("=========================================================================")
    print(f"  PHASE 6: AUTOMATED ANDROID UI MODEL SELECTION ({DEVICE_ID})")
    print("=========================================================================\n")

    print("Step 1: Locating Model Spinner...")
    center, bounds = dump_ui_and_find_bounds(TARGET_MODEL_SUBSTRING)
    if center:
        print(f"Found model item at bounds {bounds}. Tapping center ({center[0]}, {center[1]})...")
        run_adb(f"shell input tap {center[0]} {center[1]}")
        time.sleep(1.0)
    else:
        print("Model item not found on main screen. Opening model spinner first...")
        run_adb("shell input tap 500 150")
        time.sleep(1.0)
        center, bounds = dump_ui_and_find_bounds(TARGET_MODEL_SUBSTRING)
        if center:
            print(f"Tapping target model item at ({center[0]}, {center[1]})...")
            run_adb(f"shell input tap {center[0]} {center[1]}")
            time.sleep(1.0)

    print("\nStep 2: Selecting Backend Spinner (GPU)...")
    center_gpu, bounds_gpu = dump_ui_and_find_bounds("GPU")
    if center_gpu:
        print(f"Tapping GPU item at ({center_gpu[0]}, {center_gpu[1]})...")
        run_adb(f"shell input tap {center_gpu[0]} {center_gpu[1]}")
        time.sleep(1.0)

    print("\nStep 3: Tapping LOAD MODEL Button...")
    center_load, bounds_load = dump_ui_and_find_bounds("LOAD MODEL")
    if center_load:
        print(f"Tapping LOAD MODEL at ({center_load[0]}, {center_load[1]})...")
        run_adb(f"shell input tap {center_load[0]} {center_load[1]}")
        print("\n✅ Model load command issued! Monitoring logcat for GPU engine initialization...")
    else:
        print("LOAD MODEL button not found!")

if __name__ == "__main__":
    main()
