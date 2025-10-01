# src/data_gen/gen_synthetic.py
import os, random, uuid
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

AIRCRAFT = ["A320","A321","A320neo"]
BASES = ["BOM","DEL","BLR","HYD","MAA"]
RANKS = ["Captain","FirstOfficer","Cabin"]

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def gen_crew(n=150, out_file=f"{OUT_DIR}/sample_crew.csv"):
    rows = []
    for i in range(n):
        crew_id = f"C{1000+i}"
        name = f"crew_{i}"
        base = random.choice(BASES)
        rank = random.choices(RANKS, weights=[0.12,0.18,0.7])[0]
        # qualifications = aircraft types the crew is licensed to operate
        qualifications = random.sample(AIRCRAFT, k=random.randint(1, len(AIRCRAFT)))
        # preferred_aircraft_type (single preference)
        preferred_aircraft_type = random.choice(qualifications)
        license_expiry = (datetime.utcnow() + timedelta(days=random.randint(180,1800))).date().isoformat()
        preferred_days_off = random.sample(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], k=random.randint(0,3))
        max_daily_flight_minutes = random.choice([480,540,600])  # minutes
        max_weekly_flight_minutes = random.choice([2100,2400])   # minutes
        seniority_score = round(random.random(),3)
        rows.append({
            "crew_id": crew_id,
            "name": name,
            "base": base,
            "rank": rank,
            "qualifications": "|".join(qualifications),
            "preferred_aircraft_type": preferred_aircraft_type,
            "license_expiry": license_expiry,
            "max_daily_flight_minutes": max_daily_flight_minutes,
            "max_weekly_flight_minutes": max_weekly_flight_minutes,
            "preferred_days_off": "|".join(preferred_days_off),
            "leave_status": random.choices(["active","on_leave"], weights=[0.92,0.08])[0],
            "join_date": (datetime.utcnow()-timedelta(days=random.randint(30,4000))).date().isoformat(),
            "seniority_score": seniority_score,
            "past_overtime_minutes": random.randint(0,1000)
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("Wrote", out_file)
    return df

def gen_schedule(days=14, flights_per_day=40, out_file=f"{OUT_DIR}/sample_schedule.csv", start_date=None):
    rows=[]
    if start_date is None:
        start = datetime.utcnow().replace(hour=4,minute=0,second=0,microsecond=0)
    else:
        start = datetime.fromisoformat(start_date)
    for d in range(days):
        day = start + timedelta(days=d)
        for f in range(flights_per_day):
            dep = day + timedelta(minutes=30*f)
            ft = random.randint(30,240)
            arr = dep + timedelta(minutes=ft)
            dep_airport = random.choice(BASES)
            arr_airport = random.choice([b for b in BASES if b != dep_airport])
            rows.append({
                "flight_id": str(uuid.uuid4())[:8],
                "date": dep.date().isoformat(),
                "dep_time": dep.isoformat(),
                "arr_time": arr.isoformat(),
                "dep_airport": dep_airport,
                "arr_airport": arr_airport,
                "sector": f"{dep_airport}-{arr_airport}",
                "aircraft_type": random.choice(AIRCRAFT),
                "flight_time_minutes": ft,
                "required_pilots": 2,
                "required_cabin": random.randint(3,6),
                "is_international": False
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("Wrote", out_file)
    return df

def gen_historical_rosters(crew_df, schedule_df, disruptions_df=None, out_file=f"{OUT_DIR}/historical_rosters.csv"):
    rows = []
    sample_flights = schedule_df.sample(frac=0.25, random_state=1)

    # Convert disruptions to dict for quick lookup
    disruption_map = {}
    if disruptions_df is not None:
        for _, d in disruptions_df.iterrows():
            disruption_map[d["flight_id"]] = d

    for _, f in sample_flights.iterrows():
        assigned = crew_df.sample(n=random.randint(3, 8))
        for _, c in assigned.iterrows():
            disruption = disruption_map.get(f["flight_id"], None)
            rows.append({
                "roster_id": str(uuid.uuid4())[:8],
                "crew_id": c['crew_id'],
                "flight_id": f['flight_id'],
                "duty_start": f['dep_time'],
                "duty_end": f['arr_time'],
                "duty_type": "flight",
                "duty_minutes": int(f['flight_time_minutes']),
                "swap_requested": random.choice([0, 0, 0, 1]),
                "cancellation_count": random.randint(0, 2),
                "disruption_type": disruption["disruption_type"] if disruption is not None else "none",
                "delay_minutes": disruption["delay_minutes"] if disruption is not None else 0
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("Wrote", out_file)
    return df

def gen_disruptions(schedule_df, out_file=f"{OUT_DIR}/disruptions.csv"):
    rows = []
    for _, f in schedule_df.sample(frac=0.06, random_state=2).iterrows():
        cancelled = random.random() < 0.02
        delay = random.choice([0, 30, 60, 120])  # minutes
        crew_sick = random.random() < 0.01

        if cancelled:
            disruption_type = "cancel"
        elif crew_sick:
            disruption_type = "crew_sick"
        elif delay > 0:
            disruption_type = "delay"
        else:
            disruption_type = "none"

        rows.append({
            "flight_id": f['flight_id'],
            "date": f['date'],
            "disruption_type": disruption_type,
            "delay_minutes": delay if disruption_type == "delay" else 0,
            "cancelled": cancelled,
            "cause": random.choice(["weather", "technical", "crew_unavailable", "operational"]) if disruption_type != "none" else ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_file, index=False)
    print("Wrote", out_file)
    return df


if __name__ == "__main__":
    crew = gen_crew(250)
    sched = gen_schedule(1, 1)
    disruptions = gen_disruptions(sched)
    gen_historical_rosters(crew, sched, disruptions)
