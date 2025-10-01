# src/validator/compliance_validator.py
from datetime import datetime, timedelta
import pandas as pd
from src.rules.dgca_rules import DGCA_CONFIG, min_rest_after_duty

def iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def duty_minutes(start_iso: str, end_iso: str) -> int:
    s, e = iso_to_dt(start_iso), iso_to_dt(end_iso)
    return int((e - s).total_seconds() // 60)

def check_24h_limit(assignments, candidate) -> bool:
    """
    Check rolling 24h flight time limit (<= 8h).
    """
    end = iso_to_dt(candidate['duty_end'])
    window_start = end - timedelta(hours=24)
    total = 0
    for a in assignments + [candidate]:
        s = iso_to_dt(a['duty_start']); e = iso_to_dt(a['duty_end'])
        if e > window_start and s < end:
            overlap_start = max(s, window_start)
            overlap_end = min(e, end)
            total += int((overlap_end - overlap_start).total_seconds() // 60)
    return total <= DGCA_CONFIG["max_flight_minutes_24h"]

def check_weekly_cap(assignments, candidate=None) -> bool:
    """
    Check rolling 7-day cap (<= 35h).
    """
    all_assignments = assignments.copy()
    if candidate:
        all_assignments.append(candidate)
    for a in all_assignments:
        ref_end = iso_to_dt(a['duty_end'])
        week_start = ref_end - timedelta(days=7)
        tot = 0
        for b in all_assignments:
            s = iso_to_dt(b['duty_start']); e = iso_to_dt(b['duty_end'])
            if e > week_start and s < ref_end:
                overlap_start = max(s, week_start)
                overlap_end = min(e, ref_end)
                tot += int((overlap_end - overlap_start).total_seconds() // 60)
        if tot > DGCA_CONFIG["max_flight_minutes_week"]:
            return False
    return True

def check_monthly_cap(assignments, candidate=None) -> bool:
    """
    Check rolling 30-day cap (<= 125h).
    """
    all_assignments = assignments.copy()
    if candidate:
        all_assignments.append(candidate)
    for a in all_assignments:
        ref_end = iso_to_dt(a['duty_end'])
        month_start = ref_end - timedelta(days=30)
        tot = 0
        for b in all_assignments:
            s = iso_to_dt(b['duty_start']); e = iso_to_dt(b['duty_end'])
            if e > month_start and s < ref_end:
                overlap_start = max(s, month_start)
                overlap_end = min(e, ref_end)
                tot += int((overlap_end - overlap_start).total_seconds() // 60)
        if tot > DGCA_CONFIG["max_flight_minutes_month"]:
            return False
    return True

def check_fdp_limit(duty_start: str, duty_end: str) -> bool:
    """
    Check Flight Duty Period (FDP) based on day/night.
    """
    s, e = iso_to_dt(duty_start), iso_to_dt(duty_end)
    fdp_minutes = int((e - s).total_seconds() // 60)

    if DGCA_CONFIG["night_start_hour"] <= s.hour < DGCA_CONFIG["night_end_hour"]:
        return fdp_minutes <= DGCA_CONFIG["max_fdp_minutes_night"]
    return fdp_minutes <= DGCA_CONFIG["max_fdp_minutes_day"]

def check_min_rest(prev_end_iso: str, next_start_iso: str, prev_flight_minutes: int) -> bool:
    prev_end = iso_to_dt(prev_end_iso)
    next_start = iso_to_dt(next_start_iso)
    rest_min = int((next_start - prev_end).total_seconds() // 60)
    req = min_rest_after_duty(prev_flight_minutes)
    return rest_min >= req

def validate_roster(roster_path: str) -> dict:
    """
    Validate a roster CSV file against DGCA rules.
    Returns summary + violations.
    """
    df = pd.read_csv(roster_path)
    results = {"total_assignments": len(df), "violations": []}

    for idx, row in df.iterrows():
        duty_start, duty_end = row["duty_start"], row["duty_end"]
        duty_mins = duty_minutes(duty_start, duty_end)

        assignment = {"duty_start": duty_start, "duty_end": duty_end}
        assignments = df.loc[:idx - 1].to_dict("records") if idx > 0 else []

        if not check_24h_limit(assignments, assignment):
            results["violations"].append({
                "crew_id": row["crew_id"],
                "flight_id": row["flight_id"],
                "issue": "Exceeded 24h limit"
            })

        if not check_weekly_cap(assignments, assignment):
            results["violations"].append({
                "crew_id": row["crew_id"],
                "flight_id": row["flight_id"],
                "issue": "Exceeded weekly limit"
            })

        if not check_monthly_cap(assignments, assignment):
            results["violations"].append({
                "crew_id": row["crew_id"],
                "flight_id": row["flight_id"],
                "issue": "Exceeded monthly limit"
            })

        if not check_fdp_limit(duty_start, duty_end):
            results["violations"].append({
                "crew_id": row["crew_id"],
                "flight_id": row["flight_id"],
                "issue": "Exceeded FDP limit"
            })

    return results
