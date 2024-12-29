import pandas as pd

def clean_time(time_val):
    """Clean and format time values."""
    if pd.isna(time_val) or time_val == '' or time_val == 0:
        return None
    return str(int(time_val)).zfill(4)


def format_time(time_str):
    """Format time string to AM/PM format."""
    if not time_str or len(time_str) != 4:
        return None
    hours = int(time_str[:2])
    minutes = time_str[2:]
    period = "AM" if hours < 12 else "PM"
    if hours > 12:
        hours -= 12
    elif hours == 0:
        hours = 12
    return f"{hours}:{minutes} {period}"