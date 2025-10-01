# # src/api/main.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.agents.orchestrator import Orchestrator
# import os, json

# app = FastAPI()
# orch = Orchestrator()

# class SolveRequest(BaseModel):
#     crew_csv: str = "data/sample_crew.csv"
#     schedule_csv: str = "data/sample_schedule.csv"

# @app.post("/solve")
# def solve(req: SolveRequest):
#     res = orch.run_end_to_end(req.crew_csv, req.schedule_csv)
#     # save roster for audit
#     os.makedirs("out", exist_ok=True)
#     with open("out/last_roster.json","w") as f:
#         json.dump(res, f, default=str, indent=2)
#     return {"status": "ok", "assignments": len(res['assignments']), "explanation_sample": res['explanation'][:400]}

# @app.get("/explain")
# def explain():
#     # return saved explanation
#     try:
#         with open("out/last_roster.json") as f:
#             r = json.load(f)
#             return {"explanation": r.get("explanation","(none)")}
#     except FileNotFoundError:
#         return {"explanation":"No roster run yet."}





# # src/api/main.py
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from src.agents.orchestrator import Orchestrator
# import os, json

# app = FastAPI()
# orch = Orchestrator()

# # ✅ Allow frontend (React) to call API without CORS issues
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # replace with ["http://localhost:3000"] for dev
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class SolveRequest(BaseModel):
#     crew_csv: str = "data/sample_crew.csv"
#     schedule_csv: str = "data/sample_schedule.csv"
#     out_path: str = "out/roster_output.csv"

# @app.post("/solve")
# def solve(req: SolveRequest):
#     res = orch.run_pipeline(req.crew_csv, req.schedule_csv, req.out_path)

#     # Save roster for audit
#     os.makedirs("out", exist_ok=True)
#     with open("out/last_roster.json", "w") as f:
#         json.dump(res, f, default=str, indent=2)

#     return res  # Return full results (status, roster, validation, explanation)

# @app.get("/explain")
# def explain():
#     # Return saved explanation
#     try:
#         with open("out/last_roster.json") as f:
#             r = json.load(f)
#             return {"explanation": r.get("explanation", "(none)")}
#     except FileNotFoundError:
#         return {"explanation": "No roster run yet."}




# # src/api/main.py
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from src.agents.orchestrator import Orchestrator
# import os, json, shutil

# app = FastAPI()
# orch = Orchestrator()

# # ✅ Allow frontend (React) to call API without CORS issues
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # for dev: ["http://localhost:5173"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/solve")
# async def solve(
#     crew_csv: UploadFile = File(...),
#     schedule_csv: UploadFile = File(...),
# ):
#     # Save uploaded files to a temp/uploads folder
#     os.makedirs("uploads", exist_ok=True)

#     crew_path = f"uploads/{crew_csv.filename}"
#     sched_path = f"uploads/{schedule_csv.filename}"

#     with open(crew_path, "wb") as f:
#         shutil.copyfileobj(crew_csv.file, f)
#     with open(sched_path, "wb") as f:
#         shutil.copyfileobj(schedule_csv.file, f)

#     # Run pipeline
#     res = orch.run_pipeline(crew_path, sched_path, "out/roster_output.csv")

#     # Save roster for audit
#     os.makedirs("out", exist_ok=True)
#     with open("out/last_roster.json", "w") as f:
#         json.dump(res, f, default=str, indent=2)

#     return res  # Return full results (status, roster, validation, explanation)

# @app.get("/explain")
# def explain():
#     # Return saved explanation
#     try:
#         with open("out/last_roster.json") as f:
#             r = json.load(f)
#             return {"explanation": r.get("explanation", "(none)")}
#     except FileNotFoundError:
#         return {"explanation": "No roster run yet."}




# src/api/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agents.orchestrator import Orchestrator
import os, json, shutil

app = FastAPI()
orch = Orchestrator()

# ✅ Allow frontend (React) to call API without CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Roster Generation Endpoint
# -------------------------------
@app.post("/solve")
async def solve(
    crew_csv: UploadFile = File(...),
    schedule_csv: UploadFile = File(...),
):
    # Save uploaded files to a temp/uploads folder
    os.makedirs("uploads", exist_ok=True)

    crew_path = f"uploads/{crew_csv.filename}"
    sched_path = f"uploads/{schedule_csv.filename}"

    with open(crew_path, "wb") as f:
        shutil.copyfileobj(crew_csv.file, f)
    with open(sched_path, "wb") as f:
        shutil.copyfileobj(schedule_csv.file, f)

    # Run pipeline
    res = orch.run_pipeline(crew_path, sched_path, "out/roster_output.csv")

    # Save roster for audit
    os.makedirs("out", exist_ok=True)
    with open("out/last_roster.json", "w") as f:
        json.dump(res, f, default=str, indent=2)

    return res  # Return full results (status, roster, validation, explanation)


# -------------------------------
# AI Explanation Fetch
# -------------------------------
@app.get("/explain")
def explain():
    # Return saved explanation
    try:
        with open("out/last_roster.json") as f:
            r = json.load(f)
            return {"explanation": r.get("explanation", "(none)")}
    except FileNotFoundError:
        return {"explanation": "No roster run yet."}


# -------------------------------
# Disruption Handling Endpoint
# -------------------------------
class DisruptionRequest(BaseModel):
    flight_id: str | None = None
    crew_id: str | None = None
    disruption_type: str  # "cancel", "delay", "crew_unavailable"
    new_time: str | None = None  # e.g., "2025-09-10T12:00" for delays

@app.post("/disrupt")
def disrupt(req: DisruptionRequest):
    """
    Apply real-time disruption handling:
    - cancel flight
    - delay flight
    - crew unavailable
    """
    try:
        with open("out/last_roster.json") as f:
            roster_data = json.load(f)
    except FileNotFoundError:
        return {"error": "No roster found. Please generate one first."}

    roster = roster_data.get("roster", [])

    if req.disruption_type == "cancel" and req.flight_id:
        roster = [r for r in roster if r["flight_id"] != req.flight_id]

    elif req.disruption_type == "crew_unavailable" and req.crew_id:
        roster = [r for r in roster if r["crew_id"] != req.crew_id]

    elif req.disruption_type == "delay" and req.flight_id and req.new_time:
        for r in roster:
            if r["flight_id"] == req.flight_id:
                r["duty_start"] = req.new_time  # simple adjustment
                # ⚠️ you can also adjust duty_end/minutes here if needed

    roster_data["roster"] = roster

    # Save updated roster
    with open("out/last_roster.json", "w") as f:
        json.dump(roster_data, f, default=str, indent=2)

    return {
        "status": "updated",
        "disruption": req.dict(),
        "roster": roster,
    }
