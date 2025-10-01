# # src/solver/cp_scheduler.py
# """
# Constraint-based crew scheduler (OR-Tools CP-SAT).
# Accepts either CSV file paths or pandas DataFrames for crew and schedule inputs.

# Outputs:
# - assignments: list of dicts with crew_id, flight_id, duty_start, duty_end, duty_minutes, rank
# - status: CP-SAT status code (solver.StatusName(status) can be used to get readable name)
# """

# import os
# from datetime import timedelta
# from ortools.sat.python import cp_model
# import pandas as pd
# from src.rules.dgca_rules import DGCA_CONFIG


# def _ensure_df(obj):
#     """If obj is a path -> load csv into df, else assume it's already a DataFrame."""
#     if isinstance(obj, str):
#         return pd.read_csv(obj)
#     return obj.copy()


# def load_data(crew_csv_or_df, schedule_csv_or_df):
#     crew = _ensure_df(crew_csv_or_df)
#     sched = _ensure_df(schedule_csv_or_df)

#     # parse datetimes
#     sched['dep_dt'] = pd.to_datetime(sched['dep_time'])
#     sched['arr_dt'] = pd.to_datetime(sched['arr_time'])

#     # normalize qualifications list
#     crew['quals_list'] = crew.get('qualifications', '').fillna('').apply(lambda s: s.split('|') if s else [])

#     crew = crew.reset_index(drop=True)
#     sched = sched.reset_index(drop=True)
#     return crew, sched


# def build_and_solve(crew_csv_or_df, schedule_csv_or_df, preference_scores=None,
#                     max_time_seconds=30, standby_cost=50, fairness_weight=1.0, out_path=None):
#     """
#     Main solver function.
#     """
#     crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)

#     model = cp_model.CpModel()
#     C = len(crew_df)
#     F = len(sched_df)

#     # Decision vars
#     x = {}
#     s = {}
#     for i in range(C):
#         for j in range(F):
#             x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")
#             s[(i, j)] = model.NewBoolVar(f"s_{i}_{j}")

#     # Qualification constraints
#     for i, c in crew_df.iterrows():
#         quals = set(c.get('quals_list', []))
#         for j, f in sched_df.iterrows():
#             if f['aircraft_type'] not in quals:
#                 model.Add(x[(i, j)] == 0)
#                 model.Add(s[(i, j)] == 0)

#     # Coverage constraints
#     for j, f in sched_df.iterrows():
#         pilot_indices = [i for i in range(C)
#                          if crew_df.iloc[i]['rank'] in ('Captain', 'FirstOfficer')
#                          and f['aircraft_type'] in crew_df.iloc[i]['quals_list']]
#         cabin_indices = [i for i in range(C)
#                          if crew_df.iloc[i]['rank'] == 'Cabin'
#                          and f['aircraft_type'] in crew_df.iloc[i]['quals_list']]

#         req_pilots = int(f.get('required_pilots', 2))
#         req_cabin = int(f.get('required_cabin', 3))
#         if pilot_indices:
#             model.Add(sum(x[(i, j)] for i in pilot_indices) >= req_pilots)
#         if cabin_indices:
#             model.Add(sum(x[(i, j)] for i in cabin_indices) >= req_cabin)

#     # No-overlap constraints
#     for i in range(C):
#         for j in range(F):
#             for k in range(j + 1, F):
#                 s1 = sched_df.iloc[j]['dep_dt']; e1 = sched_df.iloc[j]['arr_dt']
#                 s2 = sched_df.iloc[k]['dep_dt']; e2 = sched_df.iloc[k]['arr_dt']
#                 if not (e1 <= s2 or e2 <= s1):
#                     model.Add(x[(i, j)] + x[(i, k)] <= 1)
#                     model.Add(s[(i, j)] + s[(i, k)] <= 1)

#     # Rolling 24h flight-time cap
#     MAX_24H = DGCA_CONFIG["max_flight_minutes_24h"]
#     flight_time = {j: int(sched_df.iloc[j]['flight_time_minutes']) for j in range(F)}
#     for i in range(C):
#         for j in range(F):
#             end_j = sched_df.iloc[j]['arr_dt']
#             start_window = end_j - timedelta(hours=24)
#             intersecting = []
#             for k in range(F):
#                 s_k = sched_df.iloc[k]['dep_dt']; e_k = sched_df.iloc[k]['arr_dt']
#                 if e_k > start_window and s_k < end_j:
#                     intersecting.append(k)
#             if intersecting:
#                 model.Add(sum(flight_time[k] * x[(i, k)] for k in intersecting) <= MAX_24H)

#     # Weekly cap
#     MAX_WEEK = DGCA_CONFIG["max_flight_minutes_week"]
#     crew_week_total = {}
#     for i in range(C):
#         total_minutes_var = model.NewIntVar(0, 1000000, f"total_week_{i}")
#         model.Add(total_minutes_var == sum(flight_time[j] * x[(i, j)] for j in range(F)))
#         crew_week_total[i] = total_minutes_var

#     slack_vars = []
#     for i in range(C):
#         slack = model.NewIntVar(0, 1000000, f"slack_week_{i}")
#         model.Add(crew_week_total[i] - MAX_WEEK <= slack)
#         slack_vars.append(slack)

#     # Fairness
#     total_all_minutes = sum(flight_time.values())
#     mean_est = int(total_all_minutes / max(1, C))
#     abs_diffs = []
#     for i in range(C):
#         diff = model.NewIntVar(0, 1000000, f"absdiff_{i}")
#         model.Add(diff >= crew_week_total[i] - mean_est)
#         model.Add(diff >= mean_est - crew_week_total[i])
#         abs_diffs.append(diff)

#     # Preference penalties
#     pref_penalties = []
#     if preference_scores is None:
#         preference_scores = {}
#     for i in range(C):
#         for j in range(F):
#             crew_id = crew_df.iloc[i]['crew_id']
#             flight_id = sched_df.iloc[j]['flight_id']
#             score = preference_scores.get((crew_id, flight_id), 0)
#             if score < 0:
#                 pen = model.NewIntVar(0, 1000000, f"prefpen_{i}_{j}")
#                 model.Add(pen >= int(-score * 100) * x[(i, j)])
#                 pref_penalties.append(pen)

#     # Standby penalties
#     standby_pen_vars = [s[(i, j)] for i in range(C) for j in range(F)]
#     for i in range(C):
#         for j in range(F):
#             model.Add(s[(i, j)] + x[(i, j)] <= 1)

#     # Objective
#     model.Minimize(
#         sum(slack_vars) +
#         int(fairness_weight * sum(abs_diffs)) +
#         standby_cost * sum(standby_pen_vars) +
#         sum(pref_penalties)
#     )

#     # Solve
#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = max_time_seconds
#     try:
#         solver.parameters.num_search_workers = 8
#     except Exception:
#         pass

#     status = solver.Solve(model)

#     assignments = []
#     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         for i in range(C):
#             for j in range(F):
#                 if solver.Value(x[(i, j)]) == 1:
#                     assignments.append({
#                         "crew_id": crew_df.iloc[i]['crew_id'],
#                         "flight_id": sched_df.iloc[j]['flight_id'],
#                         "duty_start": sched_df.iloc[j]['dep_time'],   # ✅ validator expects this
#                         "duty_end": sched_df.iloc[j]['arr_time'],     # ✅ validator expects this
#                         "duty_minutes": int(sched_df.iloc[j]['flight_time_minutes']),  # ✅ validator expects this
#                         "rank": crew_df.iloc[i]['rank']
#                     })
#     else:
#         return [], status

#     if out_path:
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         pd.DataFrame(assignments).to_csv(out_path, index=False)

#     return assignments, status


# def run_with_fallback(crew_csv_or_df, schedule_csv_or_df, **kwargs):
#     """
#     Wrapper that tries CP-SAT solver, falls back to greedy assignment if solver fails.
#     """
#     try:
#         return build_and_solve(crew_csv_or_df, schedule_csv_or_df, **kwargs)
#     except Exception as e:
#         print("[solver] CP-SAT failed, using greedy fallback:", e)
#         crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)

#         # --- Simple greedy fallback ---
#         assignments = []
#         for j, f in sched_df.iterrows():
#             assigned_pilots = 0
#             assigned_cabin = 0
#             for _, c in crew_df.sample(frac=1, random_state=42).iterrows():
#                 if f['aircraft_type'] not in c['quals_list']:
#                     continue
#                 if c['rank'] in ('Captain', 'FirstOfficer') and assigned_pilots < f.get('required_pilots', 2):
#                     assignments.append({
#                         "crew_id": c['crew_id'],
#                         "flight_id": f['flight_id'],
#                         "duty_start": f['dep_time'],
#                         "duty_end": f['arr_time'],
#                         "duty_minutes": int(f['flight_time_minutes']),
#                         "rank": c['rank']
#                     })
#                     assigned_pilots += 1
#                 elif c['rank'] == 'Cabin' and assigned_cabin < f.get('required_cabin', 3):
#                     assignments.append({
#                         "crew_id": c['crew_id'],
#                         "flight_id": f['flight_id'],
#                         "duty_start": f['dep_time'],
#                         "duty_end": f['arr_time'],
#                         "duty_minutes": int(f['flight_time_minutes']),
#                         "rank": c['rank']
#                     })
#                     assigned_cabin += 1
#                 if assigned_pilots >= f.get('required_pilots', 2) and assigned_cabin >= f.get('required_cabin', 3):
#                     break

#         if kwargs.get("out_path"):
#             os.makedirs(os.path.dirname(kwargs["out_path"]), exist_ok=True)
#             pd.DataFrame(assignments).to_csv(kwargs["out_path"], index=False)

#         return assignments, "GREEDY_FALLBACK"











# # src/solver/cp_scheduler.py
# """
# Constraint-based crew scheduler (OR-Tools CP-SAT) with DGCA rules embedded.

# Outputs:
# - assignments: list of dicts with crew_id, flight_id, duty_start, duty_end, duty_minutes, rank
# - status: CP-SAT status code (or "GREEDY_FALLBACK")
# """

# import os
# from datetime import timedelta
# from ortools.sat.python import cp_model
# import pandas as pd
# import numpy as np
# from src.rules.dgca_rules import DGCA_CONFIG, min_rest_after_duty


# def _ensure_df(obj):
#     if isinstance(obj, str):
#         return pd.read_csv(obj)
#     return obj.copy()


# def load_data(crew_csv_or_df, schedule_csv_or_df):
#     crew = _ensure_df(crew_csv_or_df)
#     sched = _ensure_df(schedule_csv_or_df)

#     # parse datetimes
#     sched['dep_dt'] = pd.to_datetime(sched['dep_time'])
#     sched['arr_dt'] = pd.to_datetime(sched['arr_time'])

#     # normalize qualifications list
#     crew['quals_list'] = crew.get('qualifications', '').fillna('').apply(
#         lambda s: s.split('|') if s else []
#     )

#     crew = crew.reset_index(drop=True)
#     sched = sched.reset_index(drop=True)
#     return crew, sched


# def build_and_solve(crew_csv_or_df, schedule_csv_or_df, preference_scores=None,
#                     max_time_seconds=30, standby_cost=50, fairness_weight=1.0, out_path=None):
#     """
#     Main CP-SAT solver with DGCA constraints baked in.
#     Returns (assignments, status).
#     """
#     crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)

#     model = cp_model.CpModel()
#     C = len(crew_df)
#     F = len(sched_df)

#     if C == 0 or F == 0:
#         return [], cp_model.OPTIMAL

#     # --- Precompute numeric arrays (fast)
#     dep_times = sched_df['dep_dt'].to_numpy()
#     arr_times = sched_df['arr_dt'].to_numpy()
#     flight_mins = sched_df['flight_time_minutes'].astype(int).to_list()
#     flight_ids = sched_df['flight_id'].tolist()

#     # Decision variables
#     x = {}
#     s = {}
#     for i in range(C):
#         for j in range(F):
#             x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")   # assign
#             s[(i, j)] = model.NewBoolVar(f"s_{i}_{j}")   # standby

#     # --- Qualification constraints (hard) ---
#     for i, c in crew_df.iterrows():
#         quals = set(c.get('quals_list', []))
#         for j in range(F):
#             if sched_df.loc[j, 'aircraft_type'] not in quals:
#                 model.Add(x[(i, j)] == 0)
#                 model.Add(s[(i, j)] == 0)

#     # --- Coverage constraints (per flight) ---
#     for j, f in sched_df.iterrows():
#         req_pilots = int(f.get('required_pilots', 2))
#         req_cabin = int(f.get('required_cabin', 3))

#         pilot_indices = [i for i in range(C)
#                          if crew_df.loc[i, 'rank'] in ('Captain', 'FirstOfficer')
#                          and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]
#         cabin_indices = [i for i in range(C)
#                          if crew_df.loc[i, 'rank'] == 'Cabin'
#                          and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]

#         if pilot_indices:
#             model.Add(sum(x[(i, j)] for i in pilot_indices) >= req_pilots)
#         if cabin_indices:
#             model.Add(sum(x[(i, j)] for i in cabin_indices) >= req_cabin)

#     # --- No-overlap pairwise constraints (vectorized precompute) ---
#     # overlap_matrix[j,k] = True if j overlaps k (time intervals intersect)
#     dep_arr_dep = dep_times  # alias
#     arr_arr = arr_times
#     overlap_matrix = (arr_arr[:, None] > dep_arr_dep[None, :]) & (arr_arr[None, :] > dep_arr_dep[:, None])
#     # only keep upper triangle j < k (we'll add constraints once per pair)
#     overlap_pairs = [(int(j), int(k)) for j in range(F) for k in range(j + 1, F) if overlap_matrix[j, k]]
#     # apply for every crew
#     for i in range(C):
#         for (j, k) in overlap_pairs:
#             model.Add(x[(i, j)] + x[(i, k)] <= 1)
#             model.Add(s[(i, j)] + s[(i, k)] <= 1)

#     # --- Min-rest between duties (DGCA) ---
#     # For each ordered pair (j -> k) where dep_k > arr_j, compute rest_minutes and if rest < required -> forbid pair
#     # Precompute required rest after duty j (minutes)
#     min_rest_req = [min_rest_after_duty(int(flight_mins[j])) for j in range(F)]
#     for j in range(F):
#         for k in range(F):
#             if dep_times[k] <= arr_times[j]:
#                 continue  # k not after j
#             rest_min = int((dep_times[k] - arr_times[j]).total_seconds() // 60)
#             if rest_min < min_rest_req[j]:
#                 # cannot have both flight j and k for same crew
#                 for i in range(C):
#                     model.Add(x[(i, j)] + x[(i, k)] <= 1)

#     # --- Rolling 24-hour flight-time cap (DGCA) ---
#     MAX_24H = DGCA_CONFIG["max_flight_minutes_24h"]
#     for i in range(C):
#         for j in range(F):
#             end_j = arr_times[j]
#             window_start = end_j - timedelta(hours=24)
#             intersecting = [k for k in range(F) if arr_times[k] > window_start and dep_times[k] < end_j]
#             if intersecting:
#                 # sum of flight minutes over intersecting flights assigned to crew i <= MAX_24H
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in intersecting) <= MAX_24H)

#     # --- Rolling weekly / monthly caps (rolling windows) ---
#     MAX_WEEK = DGCA_CONFIG["max_flight_minutes_week"]
#     MAX_MONTH = DGCA_CONFIG["max_flight_minutes_month"]
#     # use each flight as reference end and ensure assigned minutes in preceding window <= cap
#     for i in range(C):
#         for j in range(F):
#             ref_end = arr_times[j]
#             week_start = ref_end - timedelta(days=7)
#             month_start = ref_end - timedelta(days=30)
#             week_idx = [k for k in range(F) if arr_times[k] > week_start and dep_times[k] < ref_end]
#             month_idx = [k for k in range(F) if arr_times[k] > month_start and dep_times[k] < ref_end]
#             if week_idx:
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in week_idx) <= MAX_WEEK)
#             if month_idx:
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in month_idx) <= MAX_MONTH)

#     # --- FDP cap approximations ---
#     # For POC: enforce that any single continuous duty (sequence of overlapping/adjacent flights)
#     # does not exceed max FDP. To keep model manageable we approximate by forbidding
#     # assignment pairs that immediately create an FDP > limit when combined.
#     MAX_FDP_DAY = DGCA_CONFIG.get("max_fdp_minutes_day", 13 * 60)
#     MAX_FDP_NIGHT = DGCA_CONFIG.get("max_fdp_minutes_night", 10 * 60)
#     # Simple heuristic: if two flights j,k form a continuous duty (gap <= min_rest_req[j]? already blocked),
#     # then their combined time should not exceed FDP max (we check pairwise).
#     for j in range(F):
#         for k in range(F):
#             # consider only if j precedes k and the gap between is small (i.e., effectively same FDP)
#             if dep_times[k] <= arr_times[j]:
#                 continue
#             gap = int((dep_times[k] - arr_times[j]).total_seconds() // 60)
#             # if gap <= 60min treat as same FDP chain for approximation
#             if gap <= 60:
#                 combined = flight_mins[j] + flight_mins[k]
#                 # determine if any part of duty is night (basic check: dep hour)
#                 dep_hour_j = dep_times[j].hour
#                 dep_hour_k = dep_times[k].hour
#                 night_window = (DGCA_CONFIG["night_start_hour"], DGCA_CONFIG["night_end_hour"])
#                 any_night = (night_window[0] <= dep_hour_j < night_window[1]) or (night_window[0] <= dep_hour_k < night_window[1])
#                 max_fdp = MAX_FDP_NIGHT if any_night else MAX_FDP_DAY
#                 if combined > max_fdp:
#                     for i in range(C):
#                         model.Add(x[(i, j)] + x[(i, k)] <= 1)

#     # --- Preference penalties (soft) ---
#     if preference_scores is None:
#         preference_scores = {}
#     pref_pen_vars = []
#     for i in range(C):
#         for j in range(F):
#             score = preference_scores.get((crew_df.loc[i, 'crew_id'], sched_df.loc[j, 'flight_id']), 0)
#             # only penalize strong negative preferences (optional)
#             if score < 0:
#                 pen = model.NewIntVar(0, 1000000, f"prefpen_{i}_{j}")
#                 # scaled penalty applied if assigned
#                 model.Add(pen >= int(-score * 100) * x[(i, j)])
#                 pref_pen_vars.append(pen)

#     # --- Slack vars for weekly overtime (soft) ---
#     slack_vars = []
#     for i in range(C):
#         slack = model.NewIntVar(0, 1000000, f"slack_week_{i}")
#         # crew_week_total <= MAX_WEEK + slack
#         # compute total assigned minutes across entire schedule window for crew i
#         total_assigned = sum(int(flight_mins[j]) * x[(i, j)] for j in range(F))
#         model.Add(total_assigned - MAX_WEEK <= slack)
#         slack_vars.append(slack)

#     # --- Fairness term (approx) ---
#     total_all_minutes = sum(flight_mins)
#     mean_est = int(total_all_minutes / max(1, C))
#     abs_diff_vars = []
#     for i in range(C):
#         tot_var = sum(int(flight_mins[j]) * x[(i, j)] for j in range(F))
#         diff = model.NewIntVar(0, 10_000_000, f"absdiff_{i}")
#         # diff >= tot_var - mean_est ; diff >= mean_est - tot_var
#         # Use linear constraints: tot_var - mean_est <= diff  and mean_est - tot_var <= diff
#         model.Add(tot_var - mean_est <= diff)
#         model.Add(mean_est - tot_var <= diff)
#         abs_diff_vars.append(diff)

#     # --- Standby penalty (soft) ---
#     standby_pen_vars = [s[(i, j)] for i in range(C) for j in range(F)]

#     # link standby/assignment
#     for i in range(C):
#         for j in range(F):
#             model.Add(s[(i, j)] + x[(i, j)] <= 1)

#     # --- Objective: minimize slack + fairness + standby usage + pref penalties ---
#     objective_terms = []
#     objective_terms += slack_vars
#     objective_terms += [int(fairness_weight) * v for v in abs_diff_vars]
#     objective_terms += [standby_cost * v for v in standby_pen_vars]
#     objective_terms += pref_pen_vars

#     # Sum of all objective IntVars (if list empty, add 0)
#     if objective_terms:
#         model.Minimize(sum(objective_terms))
#     else:
#         # fallback: minimize total assignments (not really used)
#         model.Minimize(sum(x[(i, j)] for i in range(C) for j in range(F)))

#     # --- Solve ---
#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = max_time_seconds
#     try:
#         solver.parameters.num_search_workers = 8
#     except Exception:
#         pass

#     status = solver.Solve(model)

#     assignments = []
#     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         for i in range(C):
#             for j in range(F):
#                 if solver.Value(x[(i, j)]) == 1:
#                     assignments.append({
#                         "crew_id": crew_df.loc[i, 'crew_id'],
#                         "flight_id": sched_df.loc[j, 'flight_id'],
#                         "duty_start": sched_df.loc[j, 'dep_time'],
#                         "duty_end": sched_df.loc[j, 'arr_time'],
#                         "duty_minutes": int(sched_df.loc[j, 'flight_time_minutes']),
#                         "rank": crew_df.loc[i, 'rank']
#                     })
#     else:
#         # return empty to allow wrapper to fallback
#         return [], status

#     # write output
#     if out_path:
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         pd.DataFrame(assignments).to_csv(out_path, index=False)

#     return assignments, status


# def run_with_fallback(crew_csv_or_df, schedule_csv_or_df, **kwargs):
#     """
#     Try CP-SAT first; if error or infeasible, use a deterministic greedy fallback.
#     Writes same output columns.
#     """
#     try:
#         return build_and_solve(crew_csv_or_df, schedule_csv_or_df, **kwargs)
#     except Exception as e:
#         print("[solver] CP-SAT failed, using greedy fallback:", e)

#     # --- greedy deterministic fallback ---
#     crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)
#     assignments = []
#     # Sort crew by rank and low past overtime to prefer them
#     crew_df_sorted = crew_df.sort_values(['rank', 'past_overtime_minutes', 'seniority_score'])
#     for _, f in sched_df.iterrows():
#         assigned_pilots = 0
#         assigned_cabin = 0
#         for _, c in crew_df_sorted.iterrows():
#             if f['aircraft_type'] not in c['quals_list']:
#                 continue
#             if c['leave_status'] != 'active':
#                 continue
#             if c['rank'] in ('Captain', 'FirstOfficer') and assigned_pilots < int(f.get('required_pilots', 2)):
#                 assignments.append({
#                     "crew_id": c['crew_id'],
#                     "flight_id": f['flight_id'],
#                     "duty_start": f['dep_time'],
#                     "duty_end": f['arr_time'],
#                     "duty_minutes": int(f['flight_time_minutes']),
#                     "rank": c['rank']
#                 })
#                 assigned_pilots += 1
#             elif c['rank'] == 'Cabin' and assigned_cabin < int(f.get('required_cabin', 3)):
#                 assignments.append({
#                     "crew_id": c['crew_id'],
#                     "flight_id": f['flight_id'],
#                     "duty_start": f['dep_time'],
#                     "duty_end": f['arr_time'],
#                     "duty_minutes": int(f['flight_time_minutes']),
#                     "rank": c['rank']
#                 })
#                 assigned_cabin += 1
#             if assigned_pilots >= int(f.get('required_pilots', 2)) and assigned_cabin >= int(f.get('required_cabin', 3)):
#                 break

#     if kwargs.get("out_path"):
#         os.makedirs(os.path.dirname(kwargs["out_path"]), exist_ok=True)
#         pd.DataFrame(assignments).to_csv(kwargs["out_path"], index=False)

#     return assignments, "GREEDY_FALLBACK"








# """
# Constraint-based crew scheduler (OR-Tools CP-SAT) with DGCA rules embedded.

# Outputs:
# - assignments: list of dicts with crew_id, flight_id, duty_start, duty_end, duty_minutes, rank
# - status: CP-SAT status code (or "GREEDY_FALLBACK")
# """

# import os
# from datetime import timedelta
# from ortools.sat.python import cp_model
# import pandas as pd
# import numpy as np
# from src.rules.dgca_rules import DGCA_CONFIG, min_rest_after_duty


# def _ensure_df(obj):
#     if isinstance(obj, str):
#         return pd.read_csv(obj)
#     return obj.copy()


# def load_data(crew_csv_or_df, schedule_csv_or_df):
#     crew = _ensure_df(crew_csv_or_df)
#     sched = _ensure_df(schedule_csv_or_df)

#     # parse datetimes
#     sched['dep_dt'] = pd.to_datetime(sched['dep_time'])
#     sched['arr_dt'] = pd.to_datetime(sched['arr_time'])

#     # normalize qualifications list
#     crew['quals_list'] = crew.get('qualifications', '').fillna('').apply(
#         lambda s: s.split('|') if s else []
#     )

#     crew = crew.reset_index(drop=True)
#     sched = sched.reset_index(drop=True)
#     return crew, sched


# def build_and_solve(crew_csv_or_df, schedule_csv_or_df, preference_scores=None,
#                     max_time_seconds=30, standby_cost=50, fairness_weight=1.0, out_path=None):
#     """
#     Main CP-SAT solver with DGCA constraints baked in.
#     Returns (assignments, status).
#     """
#     crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)

#     model = cp_model.CpModel()
#     C = len(crew_df)
#     F = len(sched_df)

#     if C == 0 or F == 0:
#         assignments = []
#         status = cp_model.OPTIMAL
#         if out_path:
#             os.makedirs(os.path.dirname(out_path), exist_ok=True)
#             pd.DataFrame(assignments, columns=["crew_id","flight_id","duty_start","duty_end","duty_minutes","rank"]) \
#               .to_csv(out_path, index=False)
#         return assignments, status

#     # --- Precompute numeric arrays
#     dep_times = list(pd.to_datetime(sched_df['dep_dt']).to_list())
#     arr_times = list(pd.to_datetime(sched_df['arr_dt']).to_list())
#     flight_mins = sched_df['flight_time_minutes'].astype(int).to_list()

#     # Decision variables
#     x, s = {}, {}
#     for i in range(C):
#         for j in range(F):
#             x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")
#             s[(i, j)] = model.NewBoolVar(f"s_{i}_{j}")

#     # Qualification constraints
#     for i, c in crew_df.iterrows():
#         quals = set(c.get('quals_list', []))
#         for j in range(F):
#             if sched_df.loc[j, 'aircraft_type'] not in quals:
#                 model.Add(x[(i, j)] == 0)
#                 model.Add(s[(i, j)] == 0)

#     # Coverage constraints
#     for j, f in sched_df.iterrows():
#         req_pilots = int(f.get('required_pilots', 2))
#         req_cabin = int(f.get('required_cabin', 3))
#         pilot_indices = [i for i in range(C)
#                          if crew_df.loc[i, 'rank'] in ('Captain', 'FirstOfficer')
#                          and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]
#         cabin_indices = [i for i in range(C)
#                          if crew_df.loc[i, 'rank'] == 'Cabin'
#                          and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]
#         if pilot_indices:
#             model.Add(sum(x[(i, j)] for i in pilot_indices) >= req_pilots)
#         if cabin_indices:
#             model.Add(sum(x[(i, j)] for i in cabin_indices) >= req_cabin)

#     # No-overlap constraints
#     overlap_matrix = (np.array(arr_times)[:, None] > np.array(dep_times)[None, :]) & \
#                      (np.array(arr_times)[None, :] > np.array(dep_times)[:, None])
#     overlap_pairs = [(int(j), int(k)) for j in range(F) for k in range(j + 1, F) if overlap_matrix[j, k]]
#     for i in range(C):
#         for (j, k) in overlap_pairs:
#             model.Add(x[(i, j)] + x[(i, k)] <= 1)
#             model.Add(s[(i, j)] + s[(i, k)] <= 1)

#     # Min-rest constraints
#     min_rest_req = [min_rest_after_duty(int(flight_mins[j])) for j in range(F)]
#     for j in range(F):
#         for k in range(F):
#             if dep_times[k] <= arr_times[j]:
#                 continue
#             rest_min = int((dep_times[k] - arr_times[j]).total_seconds() // 60)
#             if rest_min < min_rest_req[j]:
#                 for i in range(C):
#                     model.Add(x[(i, j)] + x[(i, k)] <= 1)

#     # Rolling caps
#     MAX_24H = DGCA_CONFIG["max_flight_minutes_24h"]
#     MAX_WEEK = DGCA_CONFIG["max_flight_minutes_week"]
#     MAX_MONTH = DGCA_CONFIG["max_flight_minutes_month"]
#     for i in range(C):
#         for j in range(F):
#             end_j = arr_times[j]
#             window_start = end_j - timedelta(hours=24)
#             intersecting = [k for k in range(F) if arr_times[k] > window_start and dep_times[k] < end_j]
#             if intersecting:
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in intersecting) <= MAX_24H)
#             week_start = end_j - timedelta(days=7)
#             month_start = end_j - timedelta(days=30)
#             week_idx = [k for k in range(F) if arr_times[k] > week_start and dep_times[k] < end_j]
#             month_idx = [k for k in range(F) if arr_times[k] > month_start and dep_times[k] < end_j]
#             if week_idx:
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in week_idx) <= MAX_WEEK)
#             if month_idx:
#                 model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in month_idx) <= MAX_MONTH)

#     # Objective (simplified: fairness + standby + slack)
#     objective_terms = []
#     slack_vars = []
#     abs_diff_vars = []
#     total_all_minutes = sum(flight_mins)
#     mean_est = int(total_all_minutes / max(1, C))
#     for i in range(C):
#         tot_var = sum(int(flight_mins[j]) * x[(i, j)] for j in range(F))
#         slack = model.NewIntVar(0, 1000000, f"slack_{i}")
#         model.Add(tot_var - MAX_WEEK <= slack)
#         slack_vars.append(slack)
#         diff = model.NewIntVar(0, 10_000_000, f"absdiff_{i}")
#         model.Add(tot_var - mean_est <= diff)
#         model.Add(mean_est - tot_var <= diff)
#         abs_diff_vars.append(diff)
#     standby_pen_vars = [s[(i, j)] for i in range(C) for j in range(F)]
#     for i in range(C):
#         for j in range(F):
#             model.Add(s[(i, j)] + x[(i, j)] <= 1)
#     objective_terms += slack_vars
#     objective_terms += abs_diff_vars
#     objective_terms += standby_pen_vars
#     model.Minimize(sum(objective_terms))

#     # Solve
#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = max_time_seconds
#     try:
#         solver.parameters.num_search_workers = 8
#     except Exception:
#         pass
#     status = solver.Solve(model)

#     assignments = []
#     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         for i in range(C):
#             for j in range(F):
#                 if solver.Value(x[(i, j)]) == 1:
#                     assignments.append({
#                         "crew_id": crew_df.loc[i, 'crew_id'],
#                         "flight_id": sched_df.loc[j, 'flight_id'],
#                         "duty_start": sched_df.loc[j, 'dep_time'],
#                         "duty_end": sched_df.loc[j, 'arr_time'],
#                         "duty_minutes": int(sched_df.loc[j, 'flight_time_minutes']),
#                         "rank": crew_df.loc[i, 'rank']
#                     })

#     if out_path:
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         pd.DataFrame(assignments, columns=["crew_id","flight_id","duty_start","duty_end","duty_minutes","rank"]) \
#           .to_csv(out_path, index=False)

#     return assignments, status


# def run_with_fallback(crew_csv_or_df, schedule_csv_or_df, **kwargs):
#     try:
#         return build_and_solve(crew_csv_or_df, schedule_csv_or_df, **kwargs)
#     except Exception as e:
#         print("[solver] CP-SAT failed, using greedy fallback:", e)

#     # greedy fallback (very simplified)
#     crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)
#     assignments = []
#     for _, f in sched_df.iterrows():
#         assigned = 0
#         for _, c in crew_df.iterrows():
#             if f['aircraft_type'] in c['quals_list'] and c['leave_status'] == 'active':
#                 assignments.append({
#                     "crew_id": c['crew_id'],
#                     "flight_id": f['flight_id'],
#                     "duty_start": f['dep_time'],
#                     "duty_end": f['arr_time'],
#                     "duty_minutes": int(f['flight_time_minutes']),
#                     "rank": c['rank']
#                 })
#                 assigned += 1
#                 if assigned >= f.get('required_pilots', 2) + f.get('required_cabin', 3):
#                     break

#     if kwargs.get("out_path"):
#         os.makedirs(os.path.dirname(kwargs["out_path"]), exist_ok=True)
#         pd.DataFrame(assignments, columns=["crew_id","flight_id","duty_start","duty_end","duty_minutes","rank"]) \
#           .to_csv(kwargs["out_path"], index=False)

#     return assignments, "GREEDY_FALLBACK"









# src/solver/cp_scheduler.py --- in case of not enough solution it will still give roster 
# for flight by violating but validator will catch that
"""
Constraint-based crew scheduler (OR-Tools CP-SAT) with stronger DGCA constraints.

Changes in this patched version:
- Removed slack variables (weekly overtime is now a hard constraint).
- Added GAP_THRESHOLD to strengthen FDP chain checks (conservative multi-flight blocking).
- Increased CP-SAT time limit default to 300s (can be passed via max_time_seconds).
- Always writes out CSV (even if empty) with expected columns.
- Keeps a greedy fallback (simple) but you can replace/extend it later.
Outputs:
- assignments: list of dicts with crew_id, flight_id, duty_start, duty_end, duty_minutes, rank
- status: CP-SAT status code (or "GREEDY_FALLBACK")
"""
import os
from datetime import timedelta
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from src.rules.dgca_rules import DGCA_CONFIG, min_rest_after_duty


def _ensure_df(obj):
    if isinstance(obj, str):
        return pd.read_csv(obj)
    return obj.copy()


def load_data(crew_csv_or_df, schedule_csv_or_df):
    crew = _ensure_df(crew_csv_or_df)
    sched = _ensure_df(schedule_csv_or_df)

    # parse datetimes
    sched['dep_dt'] = pd.to_datetime(sched['dep_time'])
    sched['arr_dt'] = pd.to_datetime(sched['arr_time'])

    # normalize qualifications list
    crew['quals_list'] = crew.get('qualifications', '').fillna('').apply(
        lambda s: s.split('|') if s else []
    )

    crew = crew.reset_index(drop=True)
    sched = sched.reset_index(drop=True)
    return crew, sched


def build_and_solve(crew_csv_or_df, schedule_csv_or_df, preference_scores=None,
                    max_time_seconds=300, standby_cost=50, fairness_weight=1.0, out_path=None):
    """
    Main CP-SAT solver with DGCA constraints baked in.
    Returns (assignments, status).
    """
    crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)

    model = cp_model.CpModel()
    C = len(crew_df)
    F = len(sched_df)

    # Ensure empty outputs are written if nothing to do
    if C == 0 or F == 0:
        assignments = []
        status = cp_model.OPTIMAL
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            pd.DataFrame(assignments, columns=["crew_id", "flight_id", "duty_start", "duty_end", "duty_minutes", "rank"]) \
                .to_csv(out_path, index=False)
        return assignments, status

    # --- Precompute numeric arrays (use pandas Timestamp objects so .total_seconds() works) ---
    dep_times = list(pd.to_datetime(sched_df['dep_dt']).to_list())
    arr_times = list(pd.to_datetime(sched_df['arr_dt']).to_list())
    flight_mins = sched_df['flight_time_minutes'].astype(int).to_list()

    # Decision variables
    x = {}
    s = {}
    for i in range(C):
        for j in range(F):
            x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")
            s[(i, j)] = model.NewBoolVar(f"s_{i}_{j}")

    # --- Qualification constraints (hard) ---
    for i, c in crew_df.iterrows():
        quals = set(c.get('quals_list', []))
        for j in range(F):
            if sched_df.loc[j, 'aircraft_type'] not in quals:
                model.Add(x[(i, j)] == 0)
                model.Add(s[(i, j)] == 0)

    # --- Coverage constraints (per flight) ---
    for j, f in sched_df.iterrows():
        req_pilots = int(f.get('required_pilots', 2))
        req_cabin = int(f.get('required_cabin', 3))
        pilot_indices = [i for i in range(C)
                         if crew_df.loc[i, 'rank'] in ('Captain', 'FirstOfficer')
                         and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]
        cabin_indices = [i for i in range(C)
                         if crew_df.loc[i, 'rank'] == 'Cabin'
                         and f['aircraft_type'] in crew_df.loc[i, 'quals_list']]
        if pilot_indices:
            model.Add(sum(x[(i, j)] for i in pilot_indices) >= req_pilots)
        if cabin_indices:
            model.Add(sum(x[(i, j)] for i in cabin_indices) >= req_cabin)

    # --- No-overlap constraints (pairwise) ---
    overlap_matrix = (np.array(arr_times)[:, None] > np.array(dep_times)[None, :]) & \
                     (np.array(arr_times)[None, :] > np.array(dep_times)[:, None])
    overlap_pairs = [(int(j), int(k)) for j in range(F) for k in range(j + 1, F) if overlap_matrix[j, k]]
    for i in range(C):
        for (j, k) in overlap_pairs:
            model.Add(x[(i, j)] + x[(i, k)] <= 1)
            model.Add(s[(i, j)] + s[(i, k)] <= 1)

    # --- Min-rest constraints (pairwise prohibit if insufficient rest) ---
    min_rest_req = [min_rest_after_duty(int(flight_mins[j])) for j in range(F)]
    for j in range(F):
        for k in range(F):
            if dep_times[k] <= arr_times[j]:
                continue
            rest_min = int((dep_times[k] - arr_times[j]).total_seconds() // 60)
            if rest_min < min_rest_req[j]:
                for i in range(C):
                    model.Add(x[(i, j)] + x[(i, k)] <= 1)

    # --- Rolling caps: 24h, weekly, monthly (hard constraints) ---
    MAX_24H = DGCA_CONFIG["max_flight_minutes_24h"]
    MAX_WEEK = DGCA_CONFIG["max_flight_minutes_week"]
    MAX_MONTH = DGCA_CONFIG["max_flight_minutes_month"]
    for i in range(C):
        for j in range(F):
            # 24-hour rolling window (reference = flight j end)
            end_j = arr_times[j]
            window_start = end_j - timedelta(hours=24)
            intersecting = [k for k in range(F) if arr_times[k] > window_start and dep_times[k] < end_j]
            if intersecting:
                model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in intersecting) <= MAX_24H)

            # weekly / monthly rolling windows (reference = flight j end)
            week_start = end_j - timedelta(days=7)
            month_start = end_j - timedelta(days=30)
            week_idx = [k for k in range(F) if arr_times[k] > week_start and dep_times[k] < end_j]
            month_idx = [k for k in range(F) if arr_times[k] > month_start and dep_times[k] < end_j]
            if week_idx:
                model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in week_idx) <= MAX_WEEK)
            if month_idx:
                model.Add(sum(int(flight_mins[k]) * x[(i, k)] for k in month_idx) <= MAX_MONTH)

    # --- FDP chain strengthening (conservative) ---
    # Treat flights separated by <= GAP_THRESHOLD minutes as part of same FDP chain.
    # If combined minutes of any such pair would exceed FDP max, forbid assigning both to same crew.
    GAP_THRESHOLD = 180  # minutes (3 hours) - conservative chain window
    MAX_FDP_DAY = DGCA_CONFIG.get("max_fdp_minutes_day", 13 * 60)
    MAX_FDP_NIGHT = DGCA_CONFIG.get("max_fdp_minutes_night", 10 * 60)
    for j in range(F):
        for k in range(F):
            if dep_times[k] <= arr_times[j]:
                continue
            gap = int((dep_times[k] - arr_times[j]).total_seconds() // 60)
            if gap <= GAP_THRESHOLD:
                combined = flight_mins[j] + flight_mins[k]
                dep_hour_j = dep_times[j].hour
                dep_hour_k = dep_times[k].hour
                night_window = (DGCA_CONFIG["night_start_hour"], DGCA_CONFIG["night_end_hour"])
                any_night = (night_window[0] <= dep_hour_j < night_window[1]) or (night_window[0] <= dep_hour_k < night_window[1])
                max_fdp = MAX_FDP_NIGHT if any_night else MAX_FDP_DAY
                if combined > max_fdp:
                    for i in range(C):
                        model.Add(x[(i, j)] + x[(i, k)] <= 1)

    # --- Preference penalties (soft) ---
    if preference_scores is None:
        preference_scores = {}
    pref_pen_vars = []
    for i in range(C):
        for j in range(F):
            score = preference_scores.get((crew_df.loc[i, 'crew_id'], sched_df.loc[j, 'flight_id']), 0)
            if score < 0:
                pen = model.NewIntVar(0, 1000000, f"prefpen_{i}_{j}")
                model.Add(pen >= int(-score * 100) * x[(i, j)])
                pref_pen_vars.append(pen)

    # --- Fairness term (approx) & objective building (NO slack) ---
    total_all_minutes = sum(flight_mins)
    mean_est = int(total_all_minutes / max(1, C))
    abs_diff_vars = []
    for i in range(C):
        tot_var = sum(int(flight_mins[j]) * x[(i, j)] for j in range(F))
        diff = model.NewIntVar(0, 10_000_000, f"absdiff_{i}")
        model.Add(tot_var - mean_est <= diff)
        model.Add(mean_est - tot_var <= diff)
        abs_diff_vars.append(diff)

    # Standby penalty vars & link standby/assignment
    standby_pen_vars = [s[(i, j)] for i in range(C) for j in range(F)]
    for i in range(C):
        for j in range(F):
            model.Add(s[(i, j)] + x[(i, j)] <= 1)

    # Build objective: minimize fairness deviations + standby usage + preference penalties
    objective_terms = []
    objective_terms += [int(fairness_weight) * v for v in abs_diff_vars]
    objective_terms += standby_pen_vars
    objective_terms += pref_pen_vars

    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        model.Minimize(sum(x[(i, j)] for i in range(C) for j in range(F)))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    try:
        solver.parameters.num_search_workers = 8
    except Exception:
        pass

    status = solver.Solve(model)

    assignments = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(C):
            for j in range(F):
                if solver.Value(x[(i, j)]) == 1:
                    assignments.append({
                        "crew_id": crew_df.loc[i, 'crew_id'],
                        "flight_id": sched_df.loc[j, 'flight_id'],
                        "duty_start": sched_df.loc[j, 'dep_time'],
                        "duty_end": sched_df.loc[j, 'arr_time'],
                        "duty_minutes": int(sched_df.loc[j, 'flight_time_minutes']),
                        "rank": crew_df.loc[i, 'rank']
                    })
    else:
        # Return empty assignments if infeasible (caller may fallback)
        assignments = []

    # Always write CSV (even if empty) so validator doesn't fail on missing file
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(assignments, columns=["crew_id", "flight_id", "duty_start", "duty_end", "duty_minutes", "rank"]) \
            .to_csv(out_path, index=False)

    return assignments, status


def run_with_fallback(crew_csv_or_df, schedule_csv_or_df, **kwargs):
    """
    Try CP-SAT first; if error or infeasible, use a deterministic greedy fallback.
    """
    try:
        return build_and_solve(crew_csv_or_df, schedule_csv_or_df, **kwargs)
    except Exception as e:
        print("[solver] CP-SAT failed, using greedy fallback:", e)

    # --- improved greedy fallback (simple DGCA-aware checks) ---
    crew_df, sched_df = load_data(crew_csv_or_df, schedule_csv_or_df)
    assignments = []
    assigned_minutes = {c: 0 for c in crew_df['crew_id'].tolist()}
    MAX_WEEK = DGCA_CONFIG["max_flight_minutes_week"]
    MAX_MONTH = DGCA_CONFIG["max_flight_minutes_month"]

    crew_df_sorted = crew_df.sort_values(['rank', 'past_overtime_minutes', 'seniority_score'])

    for _, f in sched_df.iterrows():
        assigned_pilots = 0
        assigned_cabin = 0
        for _, c in crew_df_sorted.iterrows():
            if f['aircraft_type'] not in c['quals_list']:
                continue
            if c['leave_status'] != 'active':
                continue
            cid = c['crew_id']
            future_total = assigned_minutes[cid] + int(f['flight_time_minutes'])
            # simple weekly/month check
            if future_total > MAX_WEEK or future_total > MAX_MONTH:
                continue
            # simple rest check: ensure not already assigned to an overlapping flight
            conflict = False
            for a in assignments:
                if a['crew_id'] != cid:
                    continue
                # check overlap
                a_dep = pd.to_datetime(a['duty_start'])
                a_arr = pd.to_datetime(a['duty_end'])
                f_dep = pd.to_datetime(f['dep_time'])
                f_arr = pd.to_datetime(f['arr_time'])
                if not (a_arr <= f_dep or f_arr <= a_dep):
                    conflict = True
                    break
            if conflict:
                continue

            if c['rank'] in ('Captain', 'FirstOfficer') and assigned_pilots < int(f.get('required_pilots', 2)):
                assignments.append({
                    "crew_id": cid,
                    "flight_id": f['flight_id'],
                    "duty_start": f['dep_time'],
                    "duty_end": f['arr_time'],
                    "duty_minutes": int(f['flight_time_minutes']),
                    "rank": c['rank']
                })
                assigned_pilots += 1
                assigned_minutes[cid] += int(f['flight_time_minutes'])
            elif c['rank'] == 'Cabin' and assigned_cabin < int(f.get('required_cabin', 3)):
                assignments.append({
                    "crew_id": cid,
                    "flight_id": f['flight_id'],
                    "duty_start": f['dep_time'],
                    "duty_end": f['arr_time'],
                    "duty_minutes": int(f['flight_time_minutes']),
                    "rank": c['rank']
                })
                assigned_cabin += 1
                assigned_minutes[cid] += int(f['flight_time_minutes'])
            if assigned_pilots >= int(f.get('required_pilots', 2)) and assigned_cabin >= int(f.get('required_cabin', 3)):
                break

    # Ensure CSV is written even for fallback result
    if kwargs.get("out_path"):
        os.makedirs(os.path.dirname(kwargs["out_path"]), exist_ok=True)
        pd.DataFrame(assignments, columns=["crew_id", "flight_id", "duty_start", "duty_end", "duty_minutes", "rank"]) \
            .to_csv(kwargs["out_path"], index=False)

    return assignments, "GREEDY_FALLBACK"
