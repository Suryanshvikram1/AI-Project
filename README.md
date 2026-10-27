# AI-Based Timetable Scheduler

A Streamlit project for the B.Tech AI assignment. It models timetable scheduling as a CSP and implements:
- Plain Backtracking (baseline)
- Backtracking with MRV, Degree heuristic, Least-Constraining-Value ordering and forward checking-style domain checks
- Min-Conflicts local search with random-walk plateau escape
- Simulated Annealing (bonus feature)
- Hard-constraint validation and soft penalties
- Class / teacher / room views, manual edits, CSV and Excel exports
- Repeated benchmark comparison and cost-vs-iteration plots

## Requirements
Python 3.10+ recommended.

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt or Anaconda Prompt in the project folder.
3. Create and activate an environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
4. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run:
   ```bash
   streamlit run app.py
   ```

## Files
- `app.py`: Streamlit user interface
- `scheduler.py`: CSP model, algorithms, cost evaluation, exports and benchmarks
- `data/small.json`: 3 classes, 6 teachers, 4 rooms, 5 days x 6 periods
- `data/large.json`: 8 classes, 15 teachers, 8 rooms, 5 days x 7 periods

## Dataset JSON schema
Top-level keys: `subjects`, `teachers`, `classes`, `rooms`, `time`.
Teacher fields include `subjects`, `max_weekly_load`, `unavailable_periods` (day -> list of periods), and `avoid_first_period`.
Class fields include `strength`, subject codes, and `lunch_period`.
Room fields include `capacity` and type (`classroom` or `lab`).

## Assignment mapping
- Variables/domains/constraints: `build_sessions`, `_domain`, and validation/evaluation functions.
- H1-H3 clashes: teacher/class/room overlap detection.
- H4-H6: capacity, room type and teacher availability checks.
- H7: missing required sessions are reported; sessions are generated from each class's weekly subject requirements.
- H8-H9: teacher weekly load and lunch break checks.
- Soft constraints: S1 gaps, S2 subject spread, S4 teacher first-period preference, S5 consecutive teacher lectures. Lab sessions are grouped into 2-period blocks when possible.
- Cost: `1000 * hard violations + soft penalties`.
- Experiments: use the Algorithm comparison tab and report the observed results.

## Important note about experiments
The benchmark page records actual timings and results from your machine. A search limit can cause Backtracking to return no solution; this is a legitimate result to document. Do not copy fabricated numbers into your report. Run both datasets and record the results.

## Suggested team split
1. CSP model, data and hard constraints.
2. Backtracking heuristics and Min-Conflicts / Simulated Annealing.
3. Streamlit UI, experiments, report and presentation.
All team members should understand the full project and disclose AI assistance according to the assignment instructions.
