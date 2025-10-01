# # src/agents/orchestrator.py
# """
# Agent Orchestrator for AI-Powered Crew Rostering
# Uses:
#  - SchedulerAgent (OR-Tools CP-SAT with fallback)
#  - ValidatorAgent (DGCA rules compliance)
#  - ExplainAgent (LLM via Groq API) for natural language explanations
# """

# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate

# # Import scheduler with fallback safety
# from src.solver.cp_scheduler import run_with_fallback as build_and_solve
# from src.validator.compliance_validator import validate_roster

# # Load environment variables from .env
# load_dotenv()


# class SchedulerAgent:
#     """Generates an optimized crew roster using CP-SAT solver (fallback to greedy)."""
#     def run(self, crew_csv: str, schedule_csv: str, out_path: str):
#         assignments, status = build_and_solve(crew_csv, schedule_csv, out_path=out_path)
#         print(f"[SchedulerAgent] Solver status: {status}")
#         return assignments


# class ValidatorAgent:
#     """Validates roster compliance against DGCA rules."""
#     def run(self, roster_path: str):
#         results = validate_roster(roster_path)
#         return results


# class ExplainAgent:
#     """Uses Groq LLM to explain rostering decisions in natural language."""
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         model = os.getenv("LLM_MODEL", "llama3-70b-8192")

#         if not api_key:
#             raise ValueError("❌ GROQ_API_KEY not found in .env")

#         # Initialize Groq LLM
#         self.llm = ChatGroq(api_key=api_key, model=model)

#         # Prompt template
#         self.prompt = ChatPromptTemplate.from_template(
#             """
#             You are an AI assistant for IndiGo's OCC team.
#             Explain the following roster validation results clearly for managers.
#             Use simple, professional English.

#             Validation Results:
#             {results}
#             """
#         )

#     def run(self, validation_results: dict) -> str:
#         chain = self.prompt | self.llm
#         response = chain.invoke({"results": validation_results})
#         return response.content


# class Orchestrator:
#     """Coordinates the full pipeline: schedule -> roster -> validation -> explanation."""
#     def __init__(self):
#         self.scheduler = SchedulerAgent()
#         self.validator = ValidatorAgent()
#         self.explainer = ExplainAgent()

#     def run_pipeline(self, crew_csv: str, schedule_csv: str, out_path: str):
#         print("🚀 Running Scheduler...")
#         roster = self.scheduler.run(crew_csv, schedule_csv, out_path)

#         print("✅ Validating against DGCA rules...")
#         validation_results = self.validator.run(out_path)

#         print("🧠 Generating AI explanation with Groq LLM...")
#         explanation = self.explainer.run(validation_results)

#         return {
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation
#         }


# if __name__ == "__main__":
#     orch = Orchestrator()
#     result = orch.run_pipeline(
#         crew_csv="data/sample_crew.csv",
#         schedule_csv="data/sample_schedule.csv",
#         out_path="out/roster_output.csv"
#     )
#     print("Final Explanation:\n", result["explanation"])






# """
# Agent Orchestrator for AI-Powered Crew Rostering
# Uses:
#  - SchedulerAgent (OR-Tools CP-SAT with fallback)
#  - ValidatorAgent (DGCA rules compliance)
#  - ExplainAgent (LLM via Groq API) for natural language explanations
# """

# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from src.solver.cp_scheduler import run_with_fallback as build_and_solve
# from src.validator.compliance_validator import validate_roster

# load_dotenv()


# class SchedulerAgent:
#     def run(self, crew_csv: str, schedule_csv: str, out_path: str):
#         assignments, status = build_and_solve(crew_csv, schedule_csv, out_path=out_path)
#         print(f"[SchedulerAgent] Solver status: {status}")
#         return assignments, status


# class ValidatorAgent:
#     def run(self, roster_path: str):
#         return validate_roster(roster_path)


# class ExplainAgent:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         model = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
#         if not api_key:
#             raise ValueError("❌ GROQ_API_KEY not found in .env")
#         self.llm = ChatGroq(api_key=api_key, model=model)
#         self.prompt = ChatPromptTemplate.from_template(
#             """
#             You are an AI assistant for IndiGo's OCC team.
#             Explain the following roster validation results clearly for managers.
#             Use simple, professional English.

#             Validation Results:
#             {results}
#             """
#         )

#     def run(self, validation_results: dict) -> str:
#         chain = self.prompt | self.llm
#         response = chain.invoke({"results": validation_results})
#         return response.content


# class Orchestrator:
#     def __init__(self):
#         self.scheduler = SchedulerAgent()
#         self.validator = ValidatorAgent()
#         self.explainer = ExplainAgent()

#     def run_pipeline(self, crew_csv: str, schedule_csv: str, out_path: str):
#         print("🚀 Running Scheduler...")
#         roster, status = self.scheduler.run(crew_csv, schedule_csv, out_path)

#         if not roster:
#             print("❌ No feasible roster found. Skipping validation & explanation.")
#             return {
#                 "status": status,
#                 "roster": [],
#                 "validation": None,
#                 "explanation": "No feasible roster was found. Please adjust inputs or constraints."
#             }

#         print("✅ Validating against DGCA rules...")
#         validation_results = self.validator.run(out_path)

#         print("🧠 Generating AI explanation with Groq LLM...")
#         explanation = self.explainer.run(validation_results)

#         return {
#             "status": status,
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation
#         }


# if __name__ == "__main__":
#     orch = Orchestrator()
#     result = orch.run_pipeline(
#         crew_csv="data/sample_crew.csv",
#         schedule_csv="data/sample_schedule.csv",
#         out_path="out/roster_output.csv"
#     )
#     print("Final Explanation:\n", result["explanation"])






# """
# Agent Orchestrator for AI-Powered Crew Rostering
# Uses:
#  - SchedulerAgent (OR-Tools CP-SAT with fallback)
#  - ValidatorAgent (DGCA rules compliance)
#  - ExplainAgent (LLM via Groq API) for natural language explanations
# """

# import os
# import json
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from src.solver.cp_scheduler import run_with_fallback as build_and_solve
# from src.validator.compliance_validator import validate_roster

# load_dotenv()


# class SchedulerAgent:
#     def run(self, crew_csv: str, schedule_csv: str, out_path: str):
#         assignments, status = build_and_solve(crew_csv, schedule_csv, out_path=out_path)
#         print(f"[SchedulerAgent] Solver status: {status}")
#         return assignments, status


# class ValidatorAgent:
#     def run(self, roster_path: str):
#         return validate_roster(roster_path)


# class ExplainAgent:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         model = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
#         if not api_key:
#             raise ValueError("❌ GROQ_API_KEY not found in .env")
#         self.llm = ChatGroq(api_key=api_key, model=model)
#         self.prompt = ChatPromptTemplate.from_template(
#             """
#             You are an AI assistant for IndiGo's OCC team.
#             Explain the following roster validation results clearly for managers.
#             Use simple, professional English.

#             Validation Results:
#             {results}
#             """
#         )

#     def run(self, validation_results: dict) -> str:
#         chain = self.prompt | self.llm
#         response = chain.invoke({"results": validation_results})
#         return response.content


# class Orchestrator:
#     def __init__(self):
#         self.scheduler = SchedulerAgent()
#         self.validator = ValidatorAgent()
#         self.explainer = ExplainAgent()

#     def run_pipeline(self, crew_csv: str, schedule_csv: str, out_path: str):
#         print("🚀 Running Scheduler...")
#         roster, status = self.scheduler.run(crew_csv, schedule_csv, out_path)

#         if not roster:
#             print("❌ No feasible roster found. Skipping validation & explanation.")
#             return {
#                 "status": status,
#                 "roster": [],
#                 "validation": None,
#                 "explanation": "No feasible roster was found. Please adjust inputs or constraints."
#             }

#         print("✅ Validating against DGCA rules...")
#         validation_results = self.validator.run(out_path)

#         print("🧠 Generating AI explanation with Groq LLM...")
#         explanation = self.explainer.run(validation_results)

#         return {
#             "status": status,
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation
#         }

#     # -----------------------------
#     # 🔥 NEW: Disruption Handling
#     # -----------------------------
#     def apply_disruption(self, disruption: dict, crew_csv: str, schedule_csv: str, out_path: str):
#         """
#         Apply a single disruption and re-run the scheduler on modified inputs.
#         disruption = {
#             "disruption_type": "Crew Unavailable" | "Flight Cancel" | "Flight Delay",
#             "crew_id": "...",        # for crew unavailable
#             "flight_id": "...",      # for cancel/delay
#             "new_time": "...",       # for delay (ISO string)
#         }
#         """
#         crew_df = pd.read_csv(crew_csv)
#         sched_df = pd.read_csv(schedule_csv)

#         dtype = (disruption.get("disruption_type") or "").lower()

#         if "crew" in dtype and "unavail" in dtype:
#             cid = disruption.get("crew_id")
#             crew_df.loc[crew_df["crew_id"] == cid, "leave_status"] = "on_leave"

#         elif "cancel" in dtype:
#             fid = disruption.get("flight_id")
#             sched_df = sched_df[sched_df["flight_id"] != fid].reset_index(drop=True)

#         elif "delay" in dtype:
#             fid = disruption.get("flight_id")
#             new_time = disruption.get("new_time")
#             if fid and new_time:
#                 idxs = sched_df.index[sched_df["flight_id"] == fid].tolist()
#                 if idxs:
#                     idx = idxs[0]
#                     new_dep = pd.to_datetime(new_time)
#                     ft = int(sched_df.at[idx, "flight_time_minutes"])
#                     sched_df.at[idx, "dep_time"] = new_dep.isoformat()
#                     sched_df.at[idx, "arr_time"] = (new_dep + pd.Timedelta(minutes=ft)).isoformat()

#         # Save modified copies
#         os.makedirs("out", exist_ok=True)
#         tmp_crew = "out/crew_after_disruption.csv"
#         tmp_sched = "out/sched_after_disruption.csv"
#         crew_df.to_csv(tmp_crew, index=False)
#         sched_df.to_csv(tmp_sched, index=False)

#         # Re-run pipeline
#         roster, status = self.scheduler.run(tmp_crew, tmp_sched, out_path)
#         if not roster:
#             return {
#                 "status": "infeasible",
#                 "roster": [],
#                 "disruption": disruption,
#                 "validation": None,
#                 "explanation": "No feasible roster after applying disruption."
#             }

#         validation_results = self.validator.run(out_path)
#         explanation = self.explainer.run(validation_results)

#         return {
#             "status": "updated",
#             "disruption": disruption,
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation
#         }

#     def apply_disruption_csv(self, disruption_csv: str, crew_csv: str, schedule_csv: str, out_path: str):
#         """
#         Apply multiple disruptions from a CSV (columns: disruption_type, crew_id, flight_id, new_time)
#         """
#         disruptions = pd.read_csv(disruption_csv)
#         crew_df = pd.read_csv(crew_csv)
#         sched_df = pd.read_csv(schedule_csv)

#         for _, row in disruptions.iterrows():
#             dtype = (str(row.get("disruption_type") or "")).lower()

#             if "crew" in dtype and "unavail" in dtype:
#                 crew_df.loc[crew_df["crew_id"] == row.get("crew_id"), "leave_status"] = "on_leave"

#             elif "cancel" in dtype:
#                 sched_df = sched_df[sched_df["flight_id"] != row.get("flight_id")].reset_index(drop=True)

#             elif "delay" in dtype:
#                 fid = row.get("flight_id")
#                 new_time = row.get("new_time")
#                 if fid and new_time:
#                     idxs = sched_df.index[sched_df["flight_id"] == fid].tolist()
#                     if idxs:
#                         idx = idxs[0]
#                         new_dep = pd.to_datetime(new_time)
#                         ft = int(sched_df.at[idx, "flight_time_minutes"])
#                         sched_df.at[idx, "dep_time"] = new_dep.isoformat()
#                         sched_df.at[idx, "arr_time"] = (new_dep + pd.Timedelta(minutes=ft)).isoformat()

#         # Save modified copies
#         os.makedirs("out", exist_ok=True)
#         tmp_crew = "out/crew_after_bulk_disruption.csv"
#         tmp_sched = "out/sched_after_bulk_disruption.csv"
#         crew_df.to_csv(tmp_crew, index=False)
#         sched_df.to_csv(tmp_sched, index=False)

#         # Re-run pipeline
#         roster, status = self.scheduler.run(tmp_crew, tmp_sched, out_path)
#         if not roster:
#             return {
#                 "status": "infeasible",
#                 "roster": [],
#                 "validation": None,
#                 "explanation": "No feasible roster after applying bulk disruptions."
#             }

#         validation_results = self.validator.run(out_path)
#         explanation = self.explainer.run(validation_results)

#         return {
#             "status": "updated",
#             "disruptions": disruptions.to_dict(orient="records"),
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation
#         }


# if __name__ == "__main__":
#     orch = Orchestrator()
#     result = orch.run_pipeline(
#         crew_csv="data/sample_crew.csv",
#         schedule_csv="data/sample_schedule.csv",
#         out_path="out/roster_output.csv"
#     )
#     print("Final Explanation:\n", result["explanation"])




# """
# Agent Orchestrator for AI-Powered Crew Rostering
# Uses:
#  - SchedulerAgent (OR-Tools CP-SAT with fallback)
#  - ValidatorAgent (DGCA rules compliance)
#  - ExplainAgent (LLM via Groq API) for natural language explanations
# """

# import os, json
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from src.solver.cp_scheduler import run_with_fallback as build_and_solve
# from src.validator.compliance_validator import validate_roster

# load_dotenv()


# class SchedulerAgent:
#     def run(self, crew_csv: str, schedule_csv: str, out_path: str):
#         assignments, status = build_and_solve(crew_csv, schedule_csv, out_path=out_path)
#         print(f"[SchedulerAgent] Solver status: {status}")
#         return assignments, status


# class ValidatorAgent:
#     def run(self, roster_path: str):
#         return validate_roster(roster_path)


# class ExplainAgent:
#     def __init__(self):
#         api_key = os.getenv("GROQ_API_KEY")
#         model = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
#         if not api_key:
#             raise ValueError("❌ GROQ_API_KEY not found in .env")
#         self.llm = ChatGroq(api_key=api_key, model=model)
#         self.prompt = ChatPromptTemplate.from_template(
#             """
#             You are an AI assistant for IndiGo's OCC team.
#             Explain the following roster validation results clearly for managers.
#             Use simple, professional English.

#             Validation Results:
#             {results}
#             """
#         )

#     def run(self, validation_results: dict) -> str:
#         chain = self.prompt | self.llm
#         response = chain.invoke({"results": validation_results})
#         return response.content


# class Orchestrator:
#     def __init__(self):
#         self.scheduler = SchedulerAgent()
#         self.validator = ValidatorAgent()
#         self.explainer = ExplainAgent()

#     def run_pipeline(self, crew_csv: str, schedule_csv: str, out_path: str):
#         print("🚀 Running Scheduler...")
#         roster, status = self.scheduler.run(crew_csv, schedule_csv, out_path)

#         if not roster:
#             print("❌ No feasible roster found. Skipping validation & explanation.")
#             return {
#                 "status": status,
#                 "roster": [],
#                 "validation": None,
#                 "explanation": "No feasible roster was found. Please adjust inputs or constraints.",
#                 "meta": {
#                     "crew_csv": crew_csv,
#                     "schedule_csv": schedule_csv,
#                     "out_path": out_path,
#                 },
#             }

#         print("✅ Validating against DGCA rules...")
#         validation_results = self.validator.run(out_path)

#         print("🧠 Generating AI explanation with Groq LLM...")
#         explanation = self.explainer.run(validation_results)

#         return {
#             "status": status,
#             "roster": roster,
#             "validation": validation_results,
#             "explanation": explanation,
#             "meta": {
#                 "crew_csv": crew_csv,
#                 "schedule_csv": schedule_csv,
#                 "out_path": out_path,
#             },
#         }

#     def apply_disruption(self, disruption: dict, roster_path: str, out_path: str):
#         """
#         Apply disruption and re-run scheduler with updated crew/schedule.
#         Disruptions:
#          - cancel flight
#          - crew_unavailable
#          - delay flight
#         """
#         with open(roster_path) as f:
#             roster_data = json.load(f)

#         crew_csv = roster_data.get("meta", {}).get("crew_csv")
#         sched_csv = roster_data.get("meta", {}).get("schedule_csv")

#         if not crew_csv or not sched_csv:
#             raise ValueError("❌ Missing crew/schedule paths in roster metadata")

#         crew_df = pd.read_csv(crew_csv)
#         sched_df = pd.read_csv(sched_csv)

#         if disruption["disruption_type"] == "cancel" and disruption.get("flight_id"):
#             sched_df = sched_df[sched_df["flight_id"] != disruption["flight_id"]]

#         elif disruption["disruption_type"] == "crew_unavailable" and disruption.get("crew_id"):
#             cid = disruption["crew_id"]
#             crew_df.loc[crew_df["crew_id"] == cid, "leave_status"] = "on_leave"

#         elif disruption["disruption_type"] == "delay" and disruption.get("flight_id") and disruption.get("new_time"):
#             fid = disruption["flight_id"]
#             sched_df.loc[sched_df["flight_id"] == fid, "dep_time"] = disruption["new_time"]
#             # You could adjust arr_time too if delay cascades

#         # Save temporary modified files
#         crew_tmp = "uploads/tmp_disrupted_crew.csv"
#         sched_tmp = "uploads/tmp_disrupted_schedule.csv"
#         crew_df.to_csv(crew_tmp, index=False)
#         sched_df.to_csv(sched_tmp, index=False)

#         # Re-run pipeline
#         return self.run_pipeline(crew_tmp, sched_tmp, out_path)


# if __name__ == "__main__":
#     orch = Orchestrator()
#     result = orch.run_pipeline(
#         crew_csv="data/sample_crew.csv",
#         schedule_csv="data/sample_schedule.csv",
#         out_path="out/roster_output.csv",
#     )
#     print("Final Explanation:\n", result["explanation"])





"""
Agent Orchestrator for AI-Powered Crew Rostering
Uses:
 - SchedulerAgent (OR-Tools CP-SAT with fallback)
 - ValidatorAgent (DGCA rules compliance)
 - ExplainAgent (LLM via Groq API) for natural language explanations
"""

import os, json
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src.solver.cp_scheduler import run_with_fallback as build_and_solve
from src.validator.compliance_validator import validate_roster

load_dotenv()


class SchedulerAgent:
    def run(self, crew_csv: str, schedule_csv: str, out_path: str):
        assignments, status = build_and_solve(crew_csv, schedule_csv, out_path=out_path)
        print(f"[SchedulerAgent] Solver status: {status}")
        return assignments, status


class ValidatorAgent:
    def run(self, roster_path: str):
        return validate_roster(roster_path)


class ExplainAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env")
        self.llm = ChatGroq(api_key=api_key, model=model)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant for IndiGo's OCC team.
            Explain the following roster validation results clearly for managers.
            Use simple, professional English.

            Validation Results:
            {results}
            """
        )

    def run(self, validation_results: dict) -> str:
        chain = self.prompt | self.llm
        response = chain.invoke({"results": validation_results})
        return response.content


class Orchestrator:
    def __init__(self):
        self.scheduler = SchedulerAgent()
        self.validator = ValidatorAgent()
        self.explainer = ExplainAgent()

    def run_pipeline(self, crew_csv: str, schedule_csv: str, out_path: str):
        print("🚀 Running Scheduler...")
        roster, status = self.scheduler.run(crew_csv, schedule_csv, out_path)

        if not roster:
            print("❌ No feasible roster found. Skipping validation & explanation.")
            return {
                "status": status,
                "roster": [],
                "validation": None,
                "explanation": "No feasible roster was found. Please adjust inputs or constraints.",
                "meta": {
                    "crew_csv": crew_csv,
                    "schedule_csv": schedule_csv,
                    "out_path": out_path,
                },
            }

        print("✅ Validating against DGCA rules...")
        validation_results = self.validator.run(out_path)

        print("🧠 Generating AI explanation with Groq LLM...")
        explanation = self.explainer.run(validation_results)

        return {
            "status": status,
            "roster": roster,
            "validation": validation_results,
            "explanation": explanation,
            "meta": {
                "crew_csv": crew_csv,
                "schedule_csv": schedule_csv,
                "out_path": out_path,
            },
        }

    # -------------------------------
    # Apply disruption logic
    # -------------------------------
    def apply_disruption(self, disruption: dict, crew_csv: str, schedule_csv: str, out_path: str):
        """
        Apply disruption and re-run scheduler with updated crew/schedule.
        Disruptions:
         - cancel flight
         - crew_unavailable
         - delay flight
        """
        crew_df = pd.read_csv(crew_csv)
        sched_df = pd.read_csv(schedule_csv)

        d_type = disruption.get("disruption_type")

        if d_type == "cancel" and disruption.get("flight_id"):
            fid = disruption["flight_id"]
            print(f"⚠️ Cancelling flight {fid}")
            sched_df = sched_df[sched_df["flight_id"] != fid]

        elif d_type == "crew_unavailable" and disruption.get("crew_id"):
            cid = disruption["crew_id"]
            print(f"⚠️ Marking crew {cid} as unavailable")
            if "leave_status" not in crew_df.columns:
                crew_df["leave_status"] = "available"
            crew_df.loc[crew_df["crew_id"] == cid, "leave_status"] = "on_leave"

        elif d_type == "delay" and disruption.get("flight_id") and disruption.get("new_time"):
            fid = disruption["flight_id"]
            new_time = disruption["new_time"]
            print(f"⚠️ Delaying flight {fid} to {new_time}")
            if "dep_time" in sched_df.columns:
                sched_df.loc[sched_df["flight_id"] == fid, "dep_time"] = new_time

        # Save updated temporary files
        crew_tmp = "uploads/tmp_disrupted_crew.csv"
        sched_tmp = "uploads/tmp_disrupted_schedule.csv"
        crew_df.to_csv(crew_tmp, index=False)
        sched_df.to_csv(sched_tmp, index=False)

        # ✅ Re-run pipeline with updated inputs
        return self.run_pipeline(crew_tmp, sched_tmp, out_path)


if __name__ == "__main__":
    orch = Orchestrator()
    result = orch.run_pipeline(
        crew_csv="data/sample_crew.csv",
        schedule_csv="data/sample_schedule.csv",
        out_path="out/roster_output.csv",
    )
    print("Final Explanation:\n", result["explanation"])
