import pandas as pd
from copy import deepcopy
import json
import random
import itertools
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path(__file__).parent / "data"

def load_dataset(which="small"):
    path = DATA_DIR / f"{which}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_dataset(data):
    errors = []
    for key in ["subjects", "teachers", "classes", "rooms", "time"]:
        if key not in data:
            errors.append(f"Missing required section: {key}")
    if errors:
        return errors
    subject_codes = {s["code"] for s in data["subjects"]}
    teacher_names = {t["name"] for t in data["teachers"]}
    room_names = {r["name"] for r in data["rooms"]}
    if not data["time"].get("days") or int(data["time"].get("periods_per_day", 0)) < 1:
        errors.append("Time structure needs working days and periods_per_day.")
    for s in data["subjects"]:
        if int(s.get("weekly_hours", 0)) < 1:
            errors.append(f"Subject {s.get('code')} must have weekly_hours >= 1.")
        if s.get("type") not in ("theory", "lab"):
            errors.append(f"Subject {s.get('code')} type must be theory or lab.")
    for t in data["teachers"]:
        if not set(t.get("subjects", [])).issubset(subject_codes):
            errors.append(f"Teacher {t['name']} lists an unknown subject code.")
    for c in data["classes"]:
        if not set(c.get("subjects", [])).issubset(subject_codes):
            errors.append(f"Class {c['name']} lists an unknown subject code.")
        if c.get("lunch_period") not in range(1, int(data["time"]["periods_per_day"]) + 1):
            errors.append(f"Class {c['name']} has an invalid lunch_period.")
    for r in data["rooms"]:
        if r.get("type") not in ("classroom", "lab"):
            errors.append(f"Room {r['name']} type must be classroom or lab.")
    for c in data["classes"]:
        for code in c.get("subjects", []):
            if code not in subject_codes:
                continue
            if not any(code in t.get("subjects", []) for t in data["teachers"]):
                errors.append(f"No teacher can teach {code} (required by {c['name']}).")
            subj = next(s for s in data["subjects"] if s["code"] == code)
            if not any(r["type"] == ("lab" if subj["type"] == "lab" else "classroom") and r["capacity"] >= c["strength"] for r in data["rooms"]):
                errors.append(f"No suitable room for {c['name']} / {code}.")
    return errors

def build_sessions(data):
    subjects = {s["code"]: s for s in data["subjects"]}
    teachers = data["teachers"]
    sessions = []
    sid = 0
    for cls in data["classes"]:
        for code in cls["subjects"]:
            subj = subjects[code]
            eligible = [t for t in teachers if code in t["subjects"]]
            if not eligible:
                continue
            teacher = min(eligible, key=lambda t: sum(1 for x in sessions if x["teacher"] == t["name"]))
            # Lab weekly hours are grouped into 2-period blocks when possible.
            count = subj["weekly_hours"]
            block = 2 if subj["type"] == "lab" and count >= 2 else 1
            nblocks = count // block
            remainder = count % block
            for _ in range(nblocks):
                sessions.append({"id": sid, "class": cls["name"], "subject": code, "subject_name": subj["name"],
                                 "teacher": teacher["name"], "duration": block, "type": subj["type"],
                                 "strength": cls["strength"], "lunch_period": cls["lunch_period"]})
                sid += 1
            for _ in range(remainder):
                sessions.append({"id": sid, "class": cls["name"], "subject": code, "subject_name": subj["name"],
                                 "teacher": teacher["name"], "duration": 1, "type": subj["type"],
                                 "strength": cls["strength"], "lunch_period": cls["lunch_period"]})
                sid += 1
    return sessions

def _teacher_map(data):
    return {t["name"]: t for t in data["teachers"]}

def _room_map(data):
    return {r["name"]: r for r in data["rooms"]}

def _class_map(data):
    return {c["name"]: c for c in data["classes"]}

def _slots(data):
    return [(day, p) for day in data["time"]["days"] for p in range(1, data["time"]["periods_per_day"] + 1)]

def _domain(data, session):
    rooms = _room_map(data)
    teacher = _teacher_map(data)[session["teacher"]]
    result = []
    for day, period in _slots(data):
        if period in teacher.get("unavailable_periods", {}).get(day, []):
            continue
        if period == session["lunch_period"]:
            continue
        for room in data["rooms"]:
            needed = "lab" if session["type"] == "lab" else "classroom"
            if room["type"] != needed or room["capacity"] < session["strength"]:
                continue
            result.append((day, period, room["name"]))
    return result

def _occupied(session, assignment):
    day, period, room = assignment
    return [(day, period + offset, room) for offset in range(session["duration"])]

def _hard_conflict(data, s1, a1, s2, a2):
    if a1 is None or a2 is None:
        return False
    d1, p1, r1 = a1
    d2, p2, r2 = a2
    for off1 in range(s1["duration"]):
        for off2 in range(s2["duration"]):
            xday, xp = d1, p1 + off1
            yday, yp = d2, p2 + off2
            if xday == yday and xp == yp:
                if s1["class"] == s2["class"] or s1["teacher"] == s2["teacher"] or r1 == r2:
                    return True
    return False

def _assignment_valid(data, session, value, assignments, sessions, ignore_id=None):
    day, period, room = value
    if period + session["duration"] - 1 > data["time"]["periods_per_day"]:
        return False
    if any(period + off == session["lunch_period"] for off in range(session["duration"])):
        return False
    teacher = _teacher_map(data)[session["teacher"]]
    for off in range(session["duration"]):
        if period + off in teacher.get("unavailable_periods", {}).get(day, []):
            return False
    for other in sessions:
        if other["id"] == session["id"] or other["id"] == ignore_id:
            continue
        other_value = assignments.get(other["id"])
        if other_value and _hard_conflict(data, session, value, other, other_value):
            return False
    return True

def _domains(data, sessions):
    return {s["id"]: _domain(data, s) for s in sessions}

def solve_backtracking(data, heuristics=True, node_limit=12000, seed=42):
    random.seed(seed)
    sessions = build_sessions(data)
    domains = _domains(data, sessions)
    assignments = {}
    nodes = 0
    teacher_load = Counter()
    max_load = {t["name"]: t["max_weekly_load"] for t in data["teachers"]}
    if any(not domains[s["id"]] for s in sessions):
        return None, nodes, False
    def recurse():
        nonlocal nodes
        if len(assignments) == len(sessions):
            return True
        if nodes >= node_limit:
            return False
        unassigned = [s for s in sessions if s["id"] not in assignments]
        if heuristics:
            # MRV, then degree (number of unassigned sessions sharing class/teacher)
            def priority(s):
                valid = sum(_assignment_valid(data, s, v, assignments, sessions) for v in domains[s["id"]])
                degree = sum(1 for o in unassigned if o["id"] != s["id"] and
                             (o["class"] == s["class"] or o["teacher"] == s["teacher"]))
                return (valid, -degree)
            session = min(unassigned, key=priority)
            vals = [v for v in domains[session["id"]] if _assignment_valid(data, session, v, assignments, sessions)]
            # Least-constraining-value approximation: values with fewer neighbours at same time first
            def impact(v):
                d, p, r = v
                return sum(1 for o in unassigned if o["id"] != session["id"] and
                           (o["class"] == session["class"] or o["teacher"] == session["teacher"]) and
                           any(x[0] == d and x[1] == p for x in domains[o["id"]]))
            vals.sort(key=impact)
        else:
            session = unassigned[0]
            vals = [v for v in domains[session["id"]] if _assignment_valid(data, session, v, assignments, sessions)]
            random.shuffle(vals)
        for value in vals:
            nodes += 1
            if nodes > node_limit:
                return False
            assignments[session["id"]] = value
            teacher_load[session["teacher"]] += session["duration"]
            if teacher_load[session["teacher"]] <= max_load.get(session["teacher"], 999) and recurse():
                return True
            teacher_load[session["teacher"]] -= session["duration"]
            del assignments[session["id"]]
        return False
    success = recurse()
    return (assignments if success else None), nodes, success

def evaluate_schedule(data, assignments):
    sessions = build_sessions(data)
    by_id = {s["id"]: s for s in sessions}
    violations = []
    if assignments is None:
        return {"hard_violations": 0, "soft_penalty": 0, "cost": 0, "violations": [], "soft_details": []}
    # Basic invalid/missing assignment checks
    for s in sessions:
        if s["id"] not in assignments:
            violations.append({"Constraint": "H7", "Details": f"Missing session {s['id']} ({s['class']} {s['subject_name']})."})
    for sid, value in assignments.items():
        if sid not in by_id:
            continue
        s = by_id[sid]
        if value is None:
            continue
        day, period, room_name = value
        room = _room_map(data).get(room_name)
        teacher = _teacher_map(data).get(s["teacher"])
        if day not in data["time"]["days"] or period < 1 or period + s["duration"] - 1 > data["time"]["periods_per_day"]:
            violations.append({"Constraint": "H-time", "Details": f"Invalid time for session {sid}."})
            continue
        if room is None or room["capacity"] < s["strength"]:
            violations.append({"Constraint": "H4", "Details": f"Room capacity/room invalid for session {sid}."})
        elif room["type"] != ("lab" if s["type"] == "lab" else "classroom"):
            violations.append({"Constraint": "H5", "Details": f"Wrong room type for session {sid}."})
        if any(period + off == s["lunch_period"] for off in range(s["duration"])):
            violations.append({"Constraint": "H9", "Details": f"{s['class']} scheduled during lunch (session {sid})."})
        if teacher and any(period + off in teacher.get("unavailable_periods", {}).get(day, []) for off in range(s["duration"])):
            violations.append({"Constraint": "H6", "Details": f"{s['teacher']} unavailable for session {sid}."})
    for s1, s2 in itertools.combinations(sessions, 2):
        a1, a2 = assignments.get(s1["id"]), assignments.get(s2["id"])
        if a1 and a2 and _hard_conflict(data, s1, a1, s2, a2):
            violations.append({"Constraint": "H1/H2/H3", "Details": f"Sessions {s1['id']} and {s2['id']} overlap."})
    load = Counter()
    for s in sessions:
        if assignments.get(s["id"]):
            load[s["teacher"]] += s["duration"]
    for t in data["teachers"]:
        if load[t["name"]] > t["max_weekly_load"]:
            violations.append({"Constraint": "H8", "Details": f"{t['name']} load {load[t['name']]} exceeds {t['max_weekly_load']}."})
    # Subject weekly hours per class must match exactly by construction unless sessions missing
    soft_details = []
    soft_penalty = 0
    class_map = _class_map(data)
    # S1: gaps within each class/day
    for cls in data["classes"]:
        for day in data["time"]["days"]:
            periods = []
            for s in sessions:
                a = assignments.get(s["id"])
                if s["class"] == cls["name"] and a and a[0] == day:
                    periods.extend(range(a[1], a[1] + s["duration"]))
            if len(periods) >= 2:
                gaps = max(periods) - min(periods) + 1 - len(set(periods))
                if gaps:
                    soft_penalty += gaps
                    soft_details.append({"Soft constraint": "S1 gaps", "Details": f"{cls['name']} {day}: {gaps} gap(s)", "Penalty": gaps})
    # S2: same subject more than once/day (labs exempt)
    for cls in data["classes"]:
        for day in data["time"]["days"]:
            counts = Counter(s["subject"] for s in sessions if s["class"] == cls["name"] and
                             assignments.get(s["id"]) and assignments[s["id"]][0] == day and s["type"] != "lab")
            penalty = sum(max(0, n - 1) for n in counts.values())
            if penalty:
                soft_penalty += penalty
                soft_details.append({"Soft constraint": "S2 spread subjects", "Details": f"{cls['name']} {day}", "Penalty": penalty})
    # S4: teacher first-period preference
    for s in sessions:
        a = assignments.get(s["id"])
        if a and a[1] == 1:
            teacher = _teacher_map(data)[s["teacher"]]
            if teacher.get("avoid_first_period", False):
                soft_penalty += 1
                soft_details.append({"Soft constraint": "S4 teacher preference", "Details": f"{s['teacher']} first period", "Penalty": 1})
    # S5: >2 consecutive periods for teacher
    for t in data["teachers"]:
        for day in data["time"]["days"]:
            periods = set()
            for s in sessions:
                a = assignments.get(s["id"])
                if s["teacher"] == t["name"] and a and a[0] == day:
                    periods.update(range(a[1], a[1] + s["duration"]))
            run = 0
            for p in range(1, data["time"]["periods_per_day"] + 1):
                run = run + 1 if p in periods else 0
                if run == 3:
                    soft_penalty += 1
                    soft_details.append({"Soft constraint": "S5 consecutive lectures", "Details": f"{t['name']} {day}", "Penalty": 1})
    hard = len(violations)
    return {"hard_violations": hard, "soft_penalty": soft_penalty, "cost": 1000 * hard + soft_penalty,
            "violations": violations, "soft_details": soft_details}

def solve_min_conflicts(data, max_steps=12000, seed=42):
    random.seed(seed)
    sessions = build_sessions(data)
    domains = _domains(data, sessions)
    assignments = {}
    # Complete random initial assignment from each session's unary domain.
    for s in sessions:
        vals = domains[s["id"]]
        assignments[s["id"]] = random.choice(vals) if vals else None
    history = []
    best = deepcopy(assignments)
    best_cost = evaluate_schedule(data, best)["cost"]
    for step in range(max_steps):
        report = evaluate_schedule(data, assignments)
        history.append(report["cost"])
        if report["hard_violations"] == 0:
            # continue briefly for soft quality? stop at feasibility for speed
            return assignments, step + 1, True, history
        conflicted = []
        for s in sessions:
            one = {s["id"]: assignments.get(s["id"])}
            # Count conflicts by comparing current assignment to all others and unary checks
            if assignments.get(s["id"]) is None or not _assignment_valid(data, s, assignments[s["id"]],
                    {k:v for k,v in assignments.items() if k != s["id"]}, sessions):
                conflicted.append(s)
        if not conflicted:
            conflicted = sessions
        s = random.choice(conflicted)
        vals = domains[s["id"]]
        if not vals:
            continue
        if random.random() < 0.08:
            assignments[s["id"]] = random.choice(vals)
        else:
            scored = []
            for v in vals:
                assignments[s["id"]] = v
                scored.append((evaluate_schedule(data, assignments)["cost"], v))
            min_cost = min(x[0] for x in scored)
            assignments[s["id"]] = random.choice([v for cost, v in scored if cost == min_cost])
        current = evaluate_schedule(data, assignments)["cost"]
        if current < best_cost:
            best, best_cost = deepcopy(assignments), current
    history.append(evaluate_schedule(data, best)["cost"])
    return best, max_steps, evaluate_schedule(data, best)["hard_violations"] == 0, history

def solve_simulated_annealing(data, max_steps=12000, seed=42):
    random.seed(seed)
    sessions = build_sessions(data)
    domains = _domains(data, sessions)
    current = {s["id"]: random.choice(domains[s["id"]]) if domains[s["id"]] else None for s in sessions}
    best = deepcopy(current)
    current_cost = evaluate_schedule(data, current)["cost"]
    best_cost = current_cost
    history = []
    temperature = 100.0
    for step in range(max_steps):
        history.append(current_cost)
        sid = random.choice(list(current))
        if not domains[sid]:
            continue
        candidate = deepcopy(current)
        candidate[sid] = random.choice(domains[sid])
        cost = evaluate_schedule(data, candidate)["cost"]
        delta = cost - current_cost
        if delta <= 0 or random.random() < pow(2.71828, -delta / max(temperature, 0.01)):
            current, current_cost = candidate, cost
        if current_cost < best_cost:
            best, best_cost = deepcopy(current), current_cost
        temperature *= 0.995
        if best_cost == 0:
            break
    return best, len(history), evaluate_schedule(data, best)["hard_violations"] == 0, history

def schedule_to_dataframe(data, assignments):
    sessions = build_sessions(data)
    rows = []
    if assignments is None:
        return pd.DataFrame(columns=["Session ID", "Class", "Subject", "Teacher", "Day", "Period", "Room", "Duration"])
    subject_map = {s["code"]: s["name"] for s in data["subjects"]}
    for s in sessions:
        a = assignments.get(s["id"])
        if not a:
            continue
        rows.append({"Session ID": s["id"], "Class": s["class"], "Subject": subject_map.get(s["subject"], s["subject"]),
                     "Subject code": s["subject"], "Teacher": s["teacher"], "Day": a[0], "Period": a[1],
                     "Room": a[2], "Duration": s["duration"], "Type": s["type"]})
    return pd.DataFrame(rows).sort_values(["Day", "Period", "Class"]).reset_index(drop=True)

def move_session(data, assignments, session_id, day, period, room):
    updated = deepcopy(assignments)
    updated[int(session_id)] = (day, int(period), room)
    return updated

def run_benchmarks(data, repetitions=10, limit=5000, seed=42):
    import time
    configs = [
        ("Backtracking baseline", "bt", False),
        ("Backtracking heuristics", "bt", True),
        ("Min-Conflicts", "mc", True),
    ]
    rows = []
    histories = {}
    for name, kind, heur in configs:
        times, nodes, hard, soft, successes = [], [], [], [], 0
        reps = repetitions if kind == "mc" else 1
        for i in range(reps):
            start = time.perf_counter()
            if kind == "bt":
                result, count, success = solve_backtracking(data, heuristics=heur, node_limit=limit, seed=seed+i)
                hist = []
            else:
                result, count, success, hist = solve_min_conflicts(data, max_steps=limit, seed=seed+i)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            nodes.append(count)
            if result is not None:
                ev = evaluate_schedule(data, result)
                hard.append(ev["hard_violations"])
                soft.append(ev["soft_penalty"])
                successes += int(ev["hard_violations"] == 0)
                if hist:
                    histories[name] = hist
            else:
                hard.append(-1)
                soft.append(-1)
        rows.append({"Algorithm": name, "Runs": reps, "Avg time (s)": round(sum(times)/len(times), 4),
                     "Avg nodes/iterations": round(sum(nodes)/len(nodes), 1),
                     "Avg hard violations (-1 = no result)": round(sum(hard)/len(hard), 2),
                     "Avg soft penalty (-1 = no result)": round(sum(soft)/len(soft), 2),
                     "Success rate": f"{successes}/{reps}"})
    # Ensure the requested Min-Conflicts graph is available
    return pd.DataFrame(rows), histories
