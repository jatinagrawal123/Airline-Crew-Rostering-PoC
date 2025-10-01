# src/rules/dgca_rules.py
"""
DGCA Flight Duty Time Limitations (FDTL) & Rest Requirements
Source: docs/DGCA_REFERENCES.pdf
This module encodes official numeric values for compliance checks.
"""

from datetime import timedelta

DGCA_CONFIG = {
    # Flight time / duty period limits
    "max_flight_minutes_24h": 8 * 60,       # 8 hours in any rolling 24h
    "max_flight_minutes_week": 35 * 60,     # 35 hours in any rolling 7 days
    "max_flight_minutes_month": 125 * 60,   # 125 hours in 30 consecutive days
    "max_flight_minutes_year": 1000 * 60,   # 1000 hours in 365 consecutive days

    # Flight duty period (FDP) limits (baseline for 2-pilot crew, daytime ops)
    "max_fdp_minutes_day": 13 * 60,         # 13 hours
    "max_fdp_minutes_night": 10 * 60,       # 10 hours

    # Rest requirements
    "min_rest_minutes_after_duty_base": 10 * 60,   # 10 hours or equal to previous duty period (whichever is greater)
    "weekly_rest_minutes": 48 * 60,                # 48 consecutive hours free from duty every 7 days

    # Night duty definition (Window of Circadian Low, WOCL)
    "night_start_hour": 2,   # 02:00 local
    "night_end_hour": 6,     # 06:00 local
}

def min_rest_after_duty(flight_duty_minutes: int) -> int:
    """
    Rest required after a duty = max(10 hours, duty period length).
    """
    return max(
        DGCA_CONFIG["min_rest_minutes_after_duty_base"],
        flight_duty_minutes
    )
