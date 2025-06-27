import os
from datetime import datetime

# Define file paths
COUNTER_FILE = "counter.txt"
TIMESTAMP_FILE = "last_visit.txt"

def get_visit_count():
    """Retrieve the current visit count from the counter file."""
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            try:
                return int(f.read().strip())  # Read and convert to integer
            except ValueError:
                return 0  # Reset to 0 if file is corrupted
    return 0

def update_visit_count():
    """Increment and update the visit count."""
    count = get_visit_count() + 1
    with open(COUNTER_FILE, "w") as f:
        f.write(str(count))
    return count

def get_last_visit():
    """Retrieve the last visit timestamp."""
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, "r") as f:
            return f.read().strip()
    return "No previous visit"

def update_last_visit():
    """Update the last visit timestamp."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(now)
    return now