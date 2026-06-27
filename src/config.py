import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Factory Coordinates
FACTORIES = {
    "Lot's O' Nuts": {"Lat": 32.881893, "Lon": -111.768036},
    "Wicked Choccy's": {"Lat": 32.076176, "Lon": -81.088371},
    "Sugar Shack": {"Lat": 48.119140, "Lon": -96.181150},
    "Secret Factory": {"Lat": 41.446333, "Lon": -90.565487},
    "The Other Factory": {"Lat": 35.117500, "Lon": -89.971107}
}

# Product to Factory Mapping with Division
PRODUCT_DATA = {
    "Wonka Bar - Nutty Crunch Surprise": {"Factory": "Lot's O' Nuts", "Division": "Chocolate", "ID": "PRD-C-001"},
    "Wonka Bar - Fudge Mallows": {"Factory": "Lot's O' Nuts", "Division": "Chocolate", "ID": "PRD-C-002"},
    "Wonka Bar - Scrumdiddlyumptious": {"Factory": "Lot's O' Nuts", "Division": "Chocolate", "ID": "PRD-C-003"},
    "Wonka Bar - Milk Chocolate": {"Factory": "Wicked Choccy's", "Division": "Chocolate", "ID": "PRD-C-004"},
    "Wonka Bar - Triple Dazzle Caramel": {"Factory": "Wicked Choccy's", "Division": "Chocolate", "ID": "PRD-C-005"},
    "Laffy Taffy": {"Factory": "Sugar Shack", "Division": "Sugar", "ID": "PRD-S-001"},
    "SweeTARTS": {"Factory": "Sugar Shack", "Division": "Sugar", "ID": "PRD-S-002"},
    "Nerds": {"Factory": "Sugar Shack", "Division": "Sugar", "ID": "PRD-S-003"},
    "Fun Dip": {"Factory": "Sugar Shack", "Division": "Sugar", "ID": "PRD-S-004"},
    "Everlasting Gobstopper": {"Factory": "Secret Factory", "Division": "Sugar", "ID": "PRD-S-005"},
    "Hair Toffee": {"Factory": "The Other Factory", "Division": "Sugar", "ID": "PRD-S-006"},
    "Fizzy Lifting Drinks": {"Factory": "Sugar Shack", "Division": "Other", "ID": "PRD-O-001"},
    "Lickable Wallpaper": {"Factory": "Secret Factory", "Division": "Other", "ID": "PRD-O-002"},
    "Wonka Gum": {"Factory": "Secret Factory", "Division": "Other", "ID": "PRD-O-003"},
    "Kazookles": {"Factory": "The Other Factory", "Division": "Other", "ID": "PRD-O-004"}
}

# Regional Configuration
CITIES = [
    {"City": "New York", "State": "NY", "Lat": 40.7128, "Lon": -74.0060, "Region": "East"},
    {"City": "Los Angeles", "State": "CA", "Lat": 34.0522, "Lon": -118.2437, "Region": "West"},
    {"City": "Chicago", "State": "IL", "Lat": 41.8781, "Lon": -87.6298, "Region": "Midwest"},
    {"City": "Houston", "State": "TX", "Lat": 29.7604, "Lon": -95.3698, "Region": "South"},
    {"City": "Phoenix", "State": "AZ", "Lat": 33.4484, "Lon": -112.0740, "Region": "West"},
    {"City": "Philadelphia", "State": "PA", "Lat": 39.9526, "Lon": -75.1652, "Region": "East"},
    {"City": "San Antonio", "State": "TX", "Lat": 29.4241, "Lon": -98.4936, "Region": "South"},
    {"City": "San Diego", "State": "CA", "Lat": 32.7157, "Lon": -117.1611, "Region": "West"},
    {"City": "Dallas", "State": "TX", "Lat": 32.7767, "Lon": -96.7970, "Region": "South"},
    {"City": "San Jose", "State": "CA", "Lat": 37.3382, "Lon": -121.8863, "Region": "West"}
]

REGIONS = list(set([city["Region"] for city in CITIES]))
SHIP_MODES = ["Standard", "Express", "Same-Day"]

# Machine Learning Default Config
TARGET_COLUMN = 'Lead_Time_Days'
TEST_SIZE = 0.2
RANDOM_STATE = 42
