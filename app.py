import json
import time
import io

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scheduler import (
    load_dataset,
    validate_dataset,
    solve_backtracking,
    solve_min_conflicts,
    solve_simulated_annealing,
    evaluate_schedule,
    schedule_to_dataframe,
    move_session,
    run_benchmarks,
)

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Timetable Scheduler",
    page_icon="🗓️",
    layout="wide",
)

st.title("🗓️ AI-Based Timetable Scheduler")
st.caption(
    "Constraint Satisfaction Problem • Backtracking CSP • "
    "Min-Conflicts • Simulated Annealing"
)

# ---------------------------------------------------------
# Persistent session state
# These values are initialized only once per browser session.
# ---------------------------------------------------------
if "dataset" not in st.session_state:
    st.session_state.dataset = load_dataset("small")

if "schedule" not in st.session_state:
    st.session_state.schedule = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Small sample"

if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

if "benchmark_histories" not in st.session_state:
    st.session_state.benchmark_histories = {}

# ---------------------------------------------------------
# Sidebar: dataset selection and algorithm execution
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. Choose input data")

    sample = st.selectbox(
        "Sample dataset",
        ["Small sample", "Large sample"],
        key="sample_dataset_choice",
    )

    if st.button("Load selected sample", width="stretch"):
        selected_dataset = "small" if sample.startswith("Small") else "large"
        st.session_state.dataset = load_dataset(selected_dataset)
        st.session_state.dataset_name = sample
        st.session_state.schedule = None
        st.session_state.last_result = None
        st.session_state.benchmark_results = None
        st.session_state.benchmark_histories = {}
        st.success(f"{sample} loaded successfully.")
        st.rerun()

    st.markdown("**Or upload your own JSON dataset**")
    upload = st.file_uploader(
        "Upload JSON dataset",
        type=["json"],
        key="json_dataset_upload",
    )

    # Important: loading happens only after an explicit button click.
    # Merely changing a tab/radio/selectbox will NOT clear the schedule.
    if upload is not None:
        if st.button("Load JSON dataset", width="stretch"):
            try:
                upload.seek(0)
                candidate = json.load(upload)
                dataset_errors = validate_dataset(candidate)

                if dataset_errors:
                    st.error(
                        "Dataset validation failed:\n- "
                        + "\n- ".join(dataset_errors)
                    )
                else:
                    st.session_state.dataset = candidate
                    st.session_state.dataset_name = upload.name
                    st.session_state.schedule = None
                    st.session_state.last_result = None
                    st.session_state.benchmark_results = None
                    st.session_state.benchmark_histories = {}
                    st.success("JSON dataset loaded successfully.")
                    st.rerun()

            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON file: {exc}")
            except Exception as exc:
                st.error(f"Could not load JSON: {exc}")

    st.divider()
    st.header("2. Generate timetable")

    algorithm = st.selectbox(
        "Algorithm",
        [
            "Backtracking (baseline)",
            "Backtracking (MRV + Degree + Forward Checking)",
            "Min-Conflicts",
            "Simulated Annealing (bonus)",
        ],
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    limit = st.number_input(
        "Search/iteration limit",
        min_value=100,
        max_value=200000,
        value=12000,
        step=100,
    )

    if st.button(
        "🚀 Generate timetable",
        type="primary",
        width="stretch",
    ):
        dataset = st.session_state.dataset
        dataset_errors = validate_dataset(dataset)

        if dataset_errors:
            st.error(
                "Fix dataset errors first:\n- "
                + "\n- ".join(dataset_errors)
            )
        else:
            start_time = time.perf_counter()
            history = []

            try:
                if algorithm == "Backtracking (baseline)":
                    result, nodes, success = solve_backtracking(
                        dataset,
                        heuristics=False,
                        node_limit=int(limit),
                        seed=int(seed),
                    )

                elif algorithm == "Backtracking (MRV + Degree + Forward Checking)":
                    result, nodes, success = solve_backtracking(
                        dataset,
                        heuristics=True,
                        node_limit=int(limit),
                        seed=int(seed),
                    )

                elif algorithm == "Min-Conflicts":
                    result, nodes, success, history = solve_min_conflicts(
                        dataset,
                        max_steps=int(limit),
                        seed=int(seed),
                    )

                else:
                    result, nodes, success, history = solve_simulated_annealing(
                        dataset,
                        max_steps=int(limit),
                        seed=int(seed),
                    )

                elapsed = time.perf_counter() - start_time

                st.session_state.schedule = result
                st.session_state.last_result = {
                    "algorithm": algorithm,
                    "nodes": nodes,
                    "success": success,
                    "seconds": elapsed,
                    "history": history,
                }

                if result is None:
                    st.warning(
                        "No timetable was found within the search limit. "
                        "Try the heuristic Backtracking algorithm, increase "
                        "the limit, or inspect dataset feasibility."
                    )
                else:
                    evaluation = evaluate_schedule(dataset, result)
                    if evaluation["hard_violations"] == 0:
                        st.success("A valid timetable was found.")
                    else:
                        st.warning(
                            "A candidate timetable was generated, but it "
                            "contains hard-constraint violations."
                        )

            except Exception as exc:
                st.exception(exc)

# ---------------------------------------------------------
# Main application tabs
# ---------------------------------------------------------
tabs = st.tabs(
    [
        "📊 Dashboard",
        "🧑‍🏫 Timetable views",
        "✏️ Manual edit",
        "🧪 Algorithm comparison",
        "📁 Dataset & CSP",
    ]
)

data = st.session_state.dataset
dataset_errors = validate_dataset(data)
schedule = st.session_state.schedule

# ---------------------------------------------------------
# TAB 1: Dashboard
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset overview")

    if dataset_errors:
        st.error(
            "Dataset has issues:\n- " + "\n- ".join(dataset_errors)
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Classes", len(data.get("classes", [])))
    col2.metric("Teachers", len(data.get("teachers", [])))
    col3.metric("Rooms", len(data.get("rooms", [])))
    col4.metric("Subjects", len(data.get("subjects", [])))

    st.caption(f"Loaded dataset: **{st.session_state.dataset_name}**")

    if st.session_state.last_result is not None:
        result_info = st.session_state.last_result
        st.subheader("Latest algorithm run")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution time", f"{result_info['seconds']:.3f} s")
        m2.metric("Nodes / iterations", f"{result_info['nodes']:,}")
        m3.metric(
            "Algorithm success",
            "Yes" if result_info["success"] else "No",
        )
        m4.metric("Algorithm", result_info["algorithm"])

    if schedule is None:
        st.info("Generate a timetable from the sidebar to view results.")
    else:
        report = evaluate_schedule(data, schedule)

        st.subheader("Schedule quality")
        a, b, c = st.columns(3)
        a.metric("Hard violations", report["hard_violations"])
        b.metric("Soft penalty", report["soft_penalty"])
        c.metric("Total cost", report["cost"])

        if report["hard_violations"] == 0:
            st.success("All hard constraints are satisfied.")
        else:
            st.warning(
                "The schedule has hard-constraint violations. "
                "Review the diagnostics below."
            )

        if report["violations"]:
            st.write("**Hard-constraint diagnostics**")
            st.dataframe(
                pd.DataFrame(report["violations"]),
                width="stretch",
                hide_index=True,
            )

        if report["soft_details"]:
            with st.expander("Soft-constraint breakdown"):
                st.dataframe(
                    pd.DataFrame(report["soft_details"]),
                    width="stretch",
                    hide_index=True,
                )

        # Dashboard grid: choose one class so that simultaneous lectures
        # belonging to different classes do not overwrite each other.
        st.subheader("📅 Class timetable preview")
        class_names = [item["name"] for item in data["classes"]]

        if class_names:
            preview_class = st.selectbox(
                "Choose class for dashboard preview",
                class_names,
                key="dashboard_preview_class",
            )

            df = schedule_to_dataframe(data, schedule)
            class_df = df[df["Class"] == preview_class]

            days = data["time"]["days"]
            periods = range(
                1,
                data["time"]["periods_per_day"] + 1,
            )

            grid = pd.DataFrame(
                "",
                index=[f"Period {p}" for p in periods],
                columns=days,
            )

            for _, row in class_df.iterrows():
                cell_text = (
                    f"{row['Subject']}\n"
                    f"{row['Teacher']}\n"
                    f"{row['Room']}"
                )
                grid.loc[
                    f"Period {int(row['Period'])}",
                    row["Day"],
                ] = cell_text

            st.dataframe(
                grid,
                width="stretch",
                height=450,
            )

        # CSV and Excel exports
        df = schedule_to_dataframe(data, schedule)

        st.subheader("Export timetable")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download timetable CSV",
            data=csv_data,
            file_name="timetable.csv",
            mime="text/csv",
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(
                writer,
                index=False,
                sheet_name="Timetable",
            )
            pd.DataFrame(report["violations"]).to_excel(
                writer,
                index=False,
                sheet_name="Hard Violations",
            )
            pd.DataFrame(report["soft_details"]).to_excel(
                writer,
                index=False,
                sheet_name="Soft Penalties",
            )

        st.download_button(
            "⬇️ Download timetable Excel",
            data=excel_buffer.getvalue(),
            file_name="timetable.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if (
            st.session_state.last_result is not None
            and st.session_state.last_result.get("history")
        ):
            history = st.session_state.last_result["history"]

            st.subheader("Cost vs iterations")
            fig, ax = plt.subplots()
            ax.plot(range(1, len(history) + 1), history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            ax.set_title("Local-search cost history")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------
# TAB 2: Timetable views by class, teacher or room
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Timetable grid views")

    if schedule is None:
        st.info("Generate a timetable first.")
    else:
        view_type = st.radio(
            "View timetable by",
            ["Class", "Teacher", "Room"],
            horizontal=True,
        )

        entity_map = {
            "Class": data["classes"],
            "Teacher": data["teachers"],
            "Room": data["rooms"],
        }

        entity_names = [
            item["name"] for item in entity_map[view_type]
        ]

        if entity_names:
            selected_entity = st.selectbox(
                f"Select {view_type.lower()}",
                entity_names,
                key=f"selected_{view_type.lower()}_view",
            )

            timetable_df = schedule_to_dataframe(data, schedule)

            if view_type == "Class":
                filtered_df = timetable_df[
                    timetable_df["Class"] == selected_entity
                ]
            elif view_type == "Teacher":
                filtered_df = timetable_df[
                    timetable_df["Teacher"] == selected_entity
                ]
            else:
                filtered_df = timetable_df[
                    timetable_df["Room"] == selected_entity
                ]

            days = data["time"]["days"]
            periods = range(
                1,
                data["time"]["periods_per_day"] + 1,
            )

            grid = pd.DataFrame(
                "",
                index=[f"Period {p}" for p in periods],
                columns=days,
            )

            for _, row in filtered_df.iterrows():
                cell_text = (
                    f"{row['Subject']}\n"
                    f"Class: {row['Class']}\n"
                    f"Teacher: {row['Teacher']}\n"
                    f"Room: {row['Room']}"
                )

                grid.loc[
                    f"Period {int(row['Period'])}",
                    row["Day"],
                ] = cell_text

            st.dataframe(
                grid,
                width="stretch",
                height=450,
            )

            st.caption(
                "Empty cells indicate free periods. "
                "Lunch periods are defined separately for each class."
            )

# ---------------------------------------------------------
# TAB 3: Manual timetable editing
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Manual editing")

    if schedule is None:
        st.info("Generate a timetable first.")
    else:
        timetable_df = schedule_to_dataframe(data, schedule)

        if timetable_df.empty:
            st.warning("There are no scheduled sessions to edit.")
        else:
            session_labels = {
                int(row["Session ID"]): (
                    f"{row['Class']} — {row['Subject']} — "
                    f"{row['Day']} P{row['Period']} ({row['Room']})"
                )
                for _, row in timetable_df.iterrows()
            }

            selected_session_id = st.selectbox(
                "Choose session",
                list(session_labels.keys()),
                format_func=lambda value: session_labels[value],
            )

            new_day = st.selectbox(
                "New day",
                data["time"]["days"],
                key="manual_edit_day",
            )

            new_period = st.selectbox(
                "New period",
                list(
                    range(
                        1,
                        data["time"]["periods_per_day"] + 1,
                    )
                ),
                key="manual_edit_period",
            )

            room_names = [
                room["name"] for room in data["rooms"]
            ]

            new_room = st.selectbox(
                "New room",
                room_names,
                key="manual_edit_room",
            )

            if st.button(
                "Move session and check conflicts",
                type="primary",
            ):
                updated_schedule = move_session(
                    data,
                    schedule,
                    selected_session_id,
                    new_day,
                    int(new_period),
                    new_room,
                )

                st.session_state.schedule = updated_schedule
                updated_report = evaluate_schedule(
                    data,
                    updated_schedule,
                )

                if updated_report["hard_violations"] == 0:
                    st.success(
                        "Move applied. No hard-constraint conflicts."
                    )
                else:
                    st.error(
                        f"Move applied, but "
                        f"{updated_report['hard_violations']} "
                        "hard violation(s) were detected."
                    )
                    st.dataframe(
                        pd.DataFrame(updated_report["violations"]),
                        width="stretch",
                        hide_index=True,
                    )

                st.rerun()

# ---------------------------------------------------------
# TAB 4: Algorithm comparison and experiments
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Required algorithm comparison")

    st.write(
        "Compare baseline Backtracking, heuristic Backtracking "
        "and Min-Conflicts on the currently selected dataset."
    )

    runs = st.slider(
        "Runs per randomized algorithm",
        min_value=1,
        max_value=10,
        value=10,
    )

    benchmark_limit = st.number_input(
        "Benchmark node/iteration limit",
        min_value=100,
        max_value=50000,
        value=5000,
        step=500,
    )

    if st.button(
        "Run comparison experiment",
        type="primary",
    ):
        with st.spinner("Running algorithms..."):
            results, histories = run_benchmarks(
                data,
                repetitions=int(runs),
                limit=int(benchmark_limit),
                seed=int(seed),
            )

        st.session_state.benchmark_results = results
        st.session_state.benchmark_histories = histories

    if st.session_state.benchmark_results is not None:
        st.dataframe(
            st.session_state.benchmark_results,
            width="stretch",
            hide_index=True,
        )

        benchmark_csv = (
            st.session_state.benchmark_results
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download comparison CSV",
            data=benchmark_csv,
            file_name="algorithm_comparison.csv",
            mime="text/csv",
        )

        min_conflicts_history = (
            st.session_state.benchmark_histories.get(
                "Min-Conflicts",
                [],
            )
        )

        if min_conflicts_history:
            st.subheader("Min-Conflicts: cost vs iterations")

            fig, ax = plt.subplots()
            ax.plot(
                range(1, len(min_conflicts_history) + 1),
                min_conflicts_history,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            ax.set_title("Min-Conflicts cost vs iterations")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

# ---------------------------------------------------------
# TAB 5: Dataset and CSP explanation
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Dataset contents")

    for section in ["subjects", "teachers", "classes", "rooms"]:
        with st.expander(section.title()):
            st.dataframe(
                pd.DataFrame(data.get(section, [])),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Time structure"):
        st.json(data["time"])

    st.subheader("CSP formulation")

    st.markdown(
        """
        - **Variables:** Individual lecture or lab sessions.
        - **Domains:** Possible day, period and room combinations.
        - **H1:** A teacher cannot teach two sessions simultaneously.
        - **H2:** A class cannot attend two sessions simultaneously.
        - **H3:** A room cannot host two sessions simultaneously.
        - **H4:** Room capacity must be sufficient.
        - **H5:** Theory and lab sessions must use the correct room type.
        - **H6:** Teachers must be available in assigned slots.
        - **H7:** Required weekly subject sessions must be scheduled.
        - **H8:** Teacher weekly workload must not exceed the maximum.
        - **H9:** Class lunch periods must remain free.
        - **Cost:** `1000 × hard violations + soft penalties`.
        - **Soft constraints:** Reduce timetable gaps, spread subjects,
          respect teacher preferences and avoid excessive consecutive
          teaching periods.
        """
    )

    st.download_button(
        "Download current dataset JSON",
        data=json.dumps(data, indent=2).encode("utf-8"),
        file_name="timetable_dataset.json",
        mime="application/json",
    )
