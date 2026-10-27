import json
import time
import random
import io
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scheduler import (
    load_dataset, validate_dataset, build_sessions, solve_backtracking,
    solve_min_conflicts, solve_simulated_annealing, evaluate_schedule,
    schedule_to_dataframe, move_session, run_benchmarks
)

st.set_page_config(page_title="AI Timetable Scheduler", page_icon="🗓️", layout="wide")
st.title("🗓️ AI-Based Timetable Scheduler")
st.caption("Constraint Satisfaction Problem • Backtracking CSP • Min-Conflicts • Simulated Annealing")

if "dataset" not in st.session_state:
    st.session_state.dataset = load_dataset("small")
if "schedule" not in st.session_state:
    st.session_state.schedule = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Small sample"

with st.sidebar:
    st.header("1. Choose input data")
    sample = st.selectbox("Sample dataset", ["Small sample", "Large sample"], index=0)
    if st.button("Load selected sample", use_container_width=True):
        st.session_state.dataset = load_dataset("small" if sample.startswith("Small") else "large")
        st.session_state.dataset_name = sample
        st.session_state.schedule = None
        st.session_state.last_result = None
        st.rerun()

    upload = st.file_uploader("Or upload a JSON dataset", type=["json"])
    if upload is not None:
        try:
            candidate = json.load(upload)
            errors = validate_dataset(candidate)
            if errors:
                st.error("Dataset validation failed:\n- " + "\n- ".join(errors))
            else:
                st.session_state.dataset = candidate
                st.session_state.dataset_name = upload.name
                st.session_state.schedule = None
                st.success("JSON dataset loaded.")
        except Exception as exc:
            st.error(f"Could not read JSON: {exc}")

    st.divider()
    st.header("2. Generate timetable")
    algorithm = st.selectbox(
        "Algorithm",
        ["Backtracking (baseline)", "Backtracking (MRV + Degree + Forward Checking)",
         "Min-Conflicts", "Simulated Annealing (bonus)"]
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    limit = st.number_input("Search/iteration limit", min_value=100, max_value=200000, value=12000, step=100)
    if st.button("🚀 Generate timetable", type="primary", use_container_width=True):
        errors = validate_dataset(st.session_state.dataset)
        if errors:
            st.error("Fix dataset errors first:\n- " + "\n- ".join(errors))
        else:
            data = st.session_state.dataset
            start = time.perf_counter()
            if algorithm == "Backtracking (baseline)":
                result, nodes, success = solve_backtracking(data, heuristics=False, node_limit=int(limit), seed=int(seed))
            elif algorithm == "Backtracking (MRV + Degree + Forward Checking)":
                result, nodes, success = solve_backtracking(data, heuristics=True, node_limit=int(limit), seed=int(seed))
            elif algorithm == "Min-Conflicts":
                result, nodes, success, history = solve_min_conflicts(data, max_steps=int(limit), seed=int(seed))
            else:
                result, nodes, success, history = solve_simulated_annealing(data, max_steps=int(limit), seed=int(seed))
            elapsed = time.perf_counter() - start
            st.session_state.schedule = result
            st.session_state.last_result = {
                "algorithm": algorithm, "nodes": nodes, "success": success, "seconds": elapsed,
                "history": locals().get("history", [])
            }
            if result is not None:
                st.success("A valid timetable was found." if evaluate_schedule(data, result)["hard_violations"] == 0
                           else "A candidate timetable was produced; inspect the violations below.")
            else:
                st.warning("No timetable found within the search limit. Try heuristics, increase the limit, or inspect input feasibility.")

tabs = st.tabs(["📊 Dashboard", "🧑‍🏫 Timetable views", "✏️ Manual edit", "🧪 Algorithm comparison", "📁 Dataset & CSP"])

data = st.session_state.dataset
errors = validate_dataset(data)
with tabs[0]:
    st.subheader("Dataset overview")
    if errors:
        st.error("Dataset has issues:\n- " + "\n- ".join(errors))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", len(data.get("classes", [])))
    c2.metric("Teachers", len(data.get("teachers", [])))
    c3.metric("Rooms", len(data.get("rooms", [])))
    c4.metric("Subjects", len(data.get("subjects", [])))
    st.caption(f"Loaded dataset: **{st.session_state.dataset_name}**")
    if st.session_state.last_result:
        r = st.session_state.last_result
        st.subheader("Latest run")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution time", f"{r['seconds']:.3f} s")
        m2.metric("Nodes / iterations", f"{r['nodes']:,}")
        m3.metric("Hard violations", evaluate_schedule(data, st.session_state.schedule)["hard_violations"] if st.session_state.schedule else "—")
        m4.metric("Algorithm", r["algorithm"])
    if st.session_state.schedule is not None:
        report = evaluate_schedule(data, st.session_state.schedule)
        a, b, c = st.columns(3)
        a.metric("Hard violations", report["hard_violations"])
        b.metric("Soft penalty", report["soft_penalty"])
        c.metric("Total cost", report["cost"])
        if report["hard_violations"] == 0:
            st.success("All hard constraints are satisfied.")
        else:
            st.warning("The schedule has hard-constraint violations. Review the list below.")
        if report["violations"]:
            st.write("**Constraint diagnostics**")
            st.dataframe(pd.DataFrame(report["violations"]), use_container_width=True, hide_index=True)
        if report["soft_details"]:
            with st.expander("Soft-constraint breakdown"):
                st.dataframe(pd.DataFrame(report["soft_details"]), use_container_width=True, hide_index=True)
        df = schedule_to_dataframe(data, st.session_state.schedule)
        st.download_button("⬇️ Download timetable CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="timetable.csv", mime="text/csv")
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Timetable")
            pd.DataFrame(report["violations"]).to_excel(writer, index=False, sheet_name="Violations")
        st.download_button("⬇️ Download timetable Excel", xbuf.getvalue(),
                           file_name="timetable.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        if st.session_state.last_result and st.session_state.last_result.get("history"):
            hist = st.session_state.last_result["history"]
            st.subheader("Cost vs iterations")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(hist) + 1), hist)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            ax.set_title("Local-search cost history")
            ax.grid(True, alpha=.3)
            st.pyplot(fig)
    else:
        st.info("Generate a timetable from the sidebar to view results.")

with tabs[1]:
    st.subheader("Timetable grid views")
    if st.session_state.schedule is None:
        st.info("Generate a timetable first.")
    else:
        view = st.radio("View by", ["Class", "Teacher", "Room"], horizontal=True)
        entities = {"Class": data["classes"], "Teacher": data["teachers"], "Room": data["rooms"]}[view]
        selected = st.selectbox(f"Select {view.lower()}", [x["name"] for x in entities])
        df = schedule_to_dataframe(data, st.session_state.schedule)
        if view == "Class":
            sub = df[df["Class"] == selected]
        elif view == "Teacher":
            sub = df[df["Teacher"] == selected]
        else:
            sub = df[df["Room"] == selected]
        days = data["time"]["days"]
        periods = list(range(1, data["time"]["periods_per_day"] + 1))
        grid = pd.DataFrame("", index=[f"Period {p}" for p in periods], columns=days)
        for _, row in sub.iterrows():
            grid.loc[f"Period {int(row['Period'])}", row["Day"]] = (
                f"{row['Subject']}\\n{row['Class']}\\n{row['Teacher']}\\n{row['Room']}"
            )
        st.dataframe(grid, use_container_width=True, height=420)
        st.caption("Break and lunch slots are shown as empty; the dataset defines each class's lunch period.")

with tabs[2]:
    st.subheader("Manual editing")
    if st.session_state.schedule is None:
        st.info("Generate a timetable first.")
    else:
        df = schedule_to_dataframe(data, st.session_state.schedule)
        labels = {int(row["Session ID"]): f"{row['Class']} — {row['Subject']} — {row['Day']} P{row['Period']} ({row['Room']})"
                  for _, row in df.iterrows()}
        sid = st.selectbox("Choose session", list(labels), format_func=lambda x: labels[x])
        day = st.selectbox("New day", data["time"]["days"], key="edit_day")
        period = st.selectbox("New period", list(range(1, data["time"]["periods_per_day"] + 1)), key="edit_period")
        room_names = [r["name"] for r in data["rooms"]]
        room = st.selectbox("New room", room_names, key="edit_room")
        if st.button("Move session and check conflicts"):
            candidate = move_session(data, st.session_state.schedule, sid, day, int(period), room)
            st.session_state.schedule = candidate
            check = evaluate_schedule(data, candidate)
            if check["hard_violations"] == 0:
                st.success("Move applied. No hard-constraint conflicts.")
            else:
                st.error(f"Move applied, but {check['hard_violations']} hard violation(s) were detected.")
                st.dataframe(pd.DataFrame(check["violations"]), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Required algorithm comparison")
    st.write("Runs the three required configurations on the currently selected dataset. Randomized methods use repeated runs; results are measured, not fabricated.")
    runs = st.slider("Runs per randomized algorithm", 1, 10, 10)
    bench_limit = st.number_input("Benchmark node/iteration limit", min_value=100, max_value=50000, value=5000, step=500)
    if st.button("Run comparison experiment", type="primary"):
        with st.spinner("Running algorithms..."):
            results, histories = run_benchmarks(data, repetitions=int(runs), limit=int(bench_limit), seed=int(seed))
        st.session_state.benchmark_results = results
        st.session_state.benchmark_histories = histories
    if "benchmark_results" in st.session_state:
        st.dataframe(st.session_state.benchmark_results, use_container_width=True, hide_index=True)
        st.download_button("Download comparison CSV", st.session_state.benchmark_results.to_csv(index=False).encode("utf-8"),
                           file_name="algorithm_comparison.csv", mime="text/csv")
        histories = st.session_state.get("benchmark_histories", {})
        if histories.get("Min-Conflicts"):
            fig, ax = plt.subplots()
            ax.plot(range(1, len(histories["Min-Conflicts"]) + 1), histories["Min-Conflicts"])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            ax.set_title("Min-Conflicts cost vs iterations")
            ax.grid(True, alpha=.3)
            st.pyplot(fig)

with tabs[4]:
    st.subheader("Dataset contents")
    for key in ["subjects", "teachers", "classes", "rooms"]:
        with st.expander(key.title()):
            st.dataframe(pd.DataFrame(data.get(key, [])), use_container_width=True, hide_index=True)
    with st.expander("Time structure"):
        st.json(data["time"])
    st.subheader("CSP formulation")
    st.markdown("""
    - **Variable:** one required session, such as Class A's second Maths session.
    - **Domain:** possible day-period-room assignments.
    - **Hard constraints:** teacher/class/room clashes, room capacity/type, teacher availability,
      required weekly hours, teacher workload and class lunch breaks.
    - **Cost:** `1000 × hard violations + soft penalties`.
    - **Soft constraints implemented:** minimize gaps, spread subjects across days, keep lab blocks consecutive,
      respect teacher first-period preferences, and avoid more than two consecutive teacher lectures.
    """)
    st.download_button("Download current dataset JSON", json.dumps(data, indent=2).encode("utf-8"),
                       file_name="timetable_dataset.json", mime="application/json")
