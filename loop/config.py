"""Constants, location templates, colors, and configuration."""

from pathlib import Path

# ── LLM ────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"

# ── Game constants ─────────────────────────────────────────────────────
MAX_LOOPS = 7
DAYS_PER_LOOP = 5
SLOTS_PER_DAY = 4
TOTAL_SLOTS = DAYS_PER_LOOP * SLOTS_PER_DAY  # 20

# ── Trust ──────────────────────────────────────────────────────────────
TRUST_THRESHOLD_CASUAL = 20
TRUST_THRESHOLD_FRIENDLY = 40
TRUST_THRESHOLD_CONFIDE = 60
TRUST_THRESHOLD_VULNERABLE = 80

TRUST_DELTA_EMPATHY = 2           # "sorry", "understand", "how are you"
TRUST_DELTA_REMEMBERED = 4        # echoing character's earlier words
TRUST_DELTA_IMPOSSIBLE_EXPLAINED = 5   # impossible knowledge with explanation
TRUST_DELTA_IMPOSSIBLE_UNEXPLAINED = -2  # impossible knowledge, no explanation
TRUST_DELTA_ACCUSATION_MILD = -2  # "admit it"
TRUST_DELTA_ACCUSATION_HARSH = -5 # "liar", "guilty"
TRUST_DELTA_PRESSURE = -3         # "tell me now", "you must"

TRUST_FLOOR = -20  # Minimum trust value to prevent catastrophic spirals

# ── Follow mechanic ───────────────────────────────────────────────────
FOLLOW_BASE_DETECTION = 0.15
FOLLOW_TIER1_BONUS = 0.15
FOLLOW_HIGH_TRUST_REDUCTION = 0.15

# ── Evidence presentation ─────────────────────────────────────────────
MAX_EVIDENCE_PRESENTATIONS = 2
CONFRONT_TRUST_THRESHOLD = 40  # trust level where TIER2 NPCs cooperate

# ── Conversation limits ───────────────────────────────────────────────
MAX_CONVERSATION_EXCHANGES = 20

# ── Time-of-day color theme (Rich markup) ─────────────────────────────
TIME_COLORS = {
    "MORNING":   "yellow",
    "AFTERNOON": "white",
    "EVENING":   "#ff8c00",   # orange
    "NIGHT":     "blue",
}

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = DATA_DIR / "saves"

# ── Canonical locations ────────────────────────────────────────────────
# Each entry: (id, name, area, description, adjacent_ids, is_distant)
LOCATIONS_TEMPLATE = [
    {
        "id": "main_lodge",
        "name": "Main Lodge",
        "area": "central",
        "description": "The heart of Camp Pinehaven. A large timber building with a stone fireplace, bulletin boards, and the camp director's office.",
        "adjacent": ["dining_hall", "campfire_circle", "arts_cabin", "directors_office", "infirmary"],
        "is_distant": False,
    },
    {
        "id": "dining_hall",
        "name": "Dining Hall",
        "area": "central",
        "description": "Long wooden tables and benches fill this screened-in hall. The kitchen is visible through a serving window.",
        "adjacent": ["main_lodge", "kitchen", "campfire_circle"],
        "is_distant": False,
    },
    {
        "id": "kitchen",
        "name": "Kitchen",
        "area": "central",
        "description": "Industrial stoves, walk-in refrigerator, and storage shelves. Usually locked outside meal prep hours.",
        "adjacent": ["dining_hall"],
        "is_distant": False,
    },
    {
        "id": "campfire_circle",
        "name": "Campfire Circle",
        "area": "central",
        "description": "A ring of log benches around a fire pit. The social hub after dark.",
        "adjacent": ["main_lodge", "dining_hall", "trail_head", "cabin_row"],
        "is_distant": False,
    },
    {
        "id": "cabin_row",
        "name": "Cabin Row",
        "area": "living",
        "description": "A line of rustic cabins along a pine-shaded path. Each cabin sleeps six.",
        "adjacent": ["campfire_circle", "bathhouse", "staff_quarters"],
        "is_distant": False,
    },
    {
        "id": "staff_quarters",
        "name": "Staff Quarters",
        "area": "living",
        "description": "Slightly nicer cabins set apart from the main row. Where counselors sleep and keep personal belongings.",
        "adjacent": ["cabin_row", "maintenance_shed"],
        "is_distant": False,
    },
    {
        "id": "bathhouse",
        "name": "Bathhouse",
        "area": "living",
        "description": "Communal showers and restrooms. The pipes groan at night.",
        "adjacent": ["cabin_row"],
        "is_distant": False,
    },
    {
        "id": "arts_cabin",
        "name": "Arts & Crafts Cabin",
        "area": "activities",
        "description": "Paint-splattered tables, kilns, and shelves of supplies. Smells of turpentine and clay.",
        "adjacent": ["main_lodge", "archery_range"],
        "is_distant": False,
    },
    {
        "id": "archery_range",
        "name": "Archery Range",
        "area": "activities",
        "description": "A cleared field with hay-bale targets. Equipment is locked in a storage box between sessions.",
        "adjacent": ["arts_cabin", "trail_head"],
        "is_distant": False,
    },
    {
        "id": "trail_head",
        "name": "Trail Head",
        "area": "wilderness",
        "description": "Where the maintained grounds end and the forest trails begin. A faded map board marks the routes.",
        "adjacent": ["campfire_circle", "archery_range", "old_boathouse"],
        "is_distant": False,
    },
    {
        "id": "lake_shore",
        "name": "Lake Shore",
        "area": "waterfront",
        "description": "A sandy beach along Pine Lake. Canoes and kayaks rest on racks. A wooden dock extends into the water.",
        "adjacent": ["old_boathouse", "dock"],
        "is_distant": True,
    },
    {
        "id": "dock",
        "name": "The Dock",
        "area": "waterfront",
        "description": "A weathered wooden dock stretching into Pine Lake. Good for fishing, thinking, or quiet conversations.",
        "adjacent": ["lake_shore"],
        "is_distant": True,
    },
    {
        "id": "old_boathouse",
        "name": "Old Boathouse",
        "area": "waterfront",
        "description": "A dilapidated structure at the lake's edge. Officially off-limits, but the lock is broken.",
        "adjacent": ["trail_head", "lake_shore"],
        "is_distant": True,
    },
    {
        "id": "maintenance_shed",
        "name": "Maintenance Shed",
        "area": "service",
        "description": "Tools, spare parts, and the camp's generator. Smells of gasoline and sawdust.",
        "adjacent": ["staff_quarters", "storage_cellar"],
        "is_distant": False,
    },
    {
        "id": "storage_cellar",
        "name": "Storage Cellar",
        "area": "service",
        "description": "Below the main lodge. Old camp records, seasonal supplies, and forgotten equipment gather dust.",
        "adjacent": ["maintenance_shed"],
        "is_distant": False,
    },
    {
        "id": "directors_office",
        "name": "Director's Office",
        "area": "central",
        "description": "A cluttered office off the main lodge. Filing cabinets, a desk buried in papers, and a locked drawer.",
        "adjacent": ["main_lodge"],
        "is_distant": False,
    },
    {
        "id": "infirmary",
        "name": "Infirmary",
        "area": "central",
        "description": "A small medical station with cots, first aid supplies, and a medicine cabinet.",
        "adjacent": ["main_lodge"],
        "is_distant": False,
    },
]

# ── Slot index helpers ─────────────────────────────────────────────────
SLOT_NAMES = ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]


def slot_index(day: int, slot_name: str) -> int:
    """Convert 1-based day + slot name to 0-based schedule index."""
    return (day - 1) * SLOTS_PER_DAY + SLOT_NAMES.index(slot_name)


def day_and_slot(index: int) -> tuple[int, str]:
    """Convert 0-based schedule index to (1-based day, slot name)."""
    day = index // SLOTS_PER_DAY + 1
    slot = SLOT_NAMES[index % SLOTS_PER_DAY]
    return day, slot


# ── Difficulty settings ──────────────────────────────────────────────
DIFFICULTY_PRESETS = {
    "easy": {
        "max_loops": 9,
        "trust_reset_bonus": 20,
        "follow_base_detection": 0.10,
        "hints_enabled": True,
    },
    "normal": {
        "max_loops": MAX_LOOPS,
        "trust_reset_bonus": None,  # Use graduated system
        "follow_base_detection": FOLLOW_BASE_DETECTION,
        "hints_enabled": True,
    },
    "hard": {
        "max_loops": 5,
        "trust_reset_bonus": 5,
        "follow_base_detection": 0.25,
        "hints_enabled": False,
    },
}
