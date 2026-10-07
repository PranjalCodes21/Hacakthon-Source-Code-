"""
Tools for the Patient Win-Back Agent.
This is a plain Python module (NOT a notebook), so the agent notebook and, later,
the Databricks App can both import it. Each tool runs SQL through the `run_sql`
function it is given and returns short text for the LLM to read. Problems come
back as text starting with "ERROR" so the agent can change its plan instead of crashing.
The tools do all the math and enforce the business rules (capacity limits,
no duplicate contacts, holdout assignment). The LLM decides what to investigate and what to do.
Holdout group: a random share of patients the agent chooses (holdout_share) is recorded but
NOT contacted. Comparing their return rate with contacted patients measures the lift that
outreach actually caused. Assignment is a hash of run_id + patient_id, so it is reproducible
and the agent cannot influence it.
"""
import hashlib
import json
import math
import re
from decimal import Decimal
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]$")
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
Dimensions the agent may test as possible drop-off causes (name -> column/expression).
DIMENSIONS = {
"home_location": "home_location_id",
"last_location": "last_location_id",
"acquisition_source": "acquisition_source",
"age_band": "age_band",
"last_service_type": "last_service_type",
"last_provider_active": "CAST(last_provider_active AS STRING)",
}
REASONS = [
"provider_inactive", # their usual provider no longer works here
"never_started_care", # only ever came for an initial consultation
"clinic_capacity", # routed to a clinic with more open slots
"frequent_no_shows", # history of missed appointments
"high_value_lapsed", # valuable patient who drifted away
"general_reengagement", # none of the above stands out
]
FILTERS = {
"provider_inactive": "NOT last_provider_active",
"never_started_care": "consult_only",
"frequent_no_shows": "n_no_shows >= 2",
}

def _lit(value):
"""SQL string literal with escaping."""
if value is None:
return "NULL"
s = str(value).replace("\", "\\").replace("'", "\'")
return f"'{s}'"

def _cell(v):
if v is None:
return ""
if isinstance(v, (float, Decimal)):
return f"{float(v):.3f}".rstrip("0").rstrip(".")
return str(v)

def _fmt(rows, max_rows=40):
"""Rows (list of dicts) -> compact CSV text."""
if not rows:
return "(no rows)"
cols = list(rows[0].keys())
lines = [",".join(cols)]
for r in rows[:max_rows]:
lines.append(",".join(_cell(r[c]) for c in cols))
if len(rows) > max_rows:
lines.append(f"... {len(rows) - max_rows} more rows")
return "
".join(lines)

class WinbackTools:
def __init__(self, run_sql, catalog, schema, run_id,
lapsed_days=180, max_actions=15, capacity_share=0.02, holdout_share=0.1):
"""
run_sql: function(query: str) -> list[dict]
capacity_share: fraction of each clinic's weekly open slots this run may fill.
holdout_share: fraction of chosen patients randomly held out (recorded, not contacted).
"""
for name in (catalog, schema):
if not IDENT.match(name):
raise ValueError(f"Invalid identifier: {name}")
self.sql = run_sql
self.fq = f"{catalog}.{schema}"
self.run_id = run_id
self.lapsed_days = int(lapsed_days)
self.max_actions = int(max_actions)
self.capacity_share = float(capacity_share)
self.holdout_share = float(holdout_share)
self.used = {} # location_id -> contacts this run
self.contacts = 0
self.skips = 0
self.holdouts = 0
self.profile_calls = 0
self._budget = None
# ---------- helpers ----------
def _budgets(self):
if self._budget is None:
rows = self.sql(f"SELECT location_id, open_slots_per_week FROM {self.fq}.location_capacity")
self._budget = {
r["location_id"]: max(1, int(math.floor((r["open_slots_per_week"] or 0) * self.capacity_share)))
for r in rows
}
return self._budget
def _remaining(self, location_id):
return self._budgets().get(location_id, 0) - self.used.get(location_id, 0)
def _in_holdout(self, patient_id):
"""Deterministic random assignment: the agent cannot influence who is held out."""
h = hashlib.sha256(f"{self.run_id}:{patient_id}".encode()).hexdigest()
return int(h[:8], 16) / 0xFFFFFFFF < self.holdout_share
# ---------- tools ----------
def review_past_outcomes(self):
rows = self.sql(f"""
SELECT reason_category,
SUM(CASE WHEN COALESCE(arm, 'contacted') = 'contacted' THEN 1 ELSE 0 END) AS contacted,
SUM(CASE WHEN arm = 'holdout' THEN 1 ELSE 0 END) AS held_out,
CAST(AVG(CASE WHEN COALESCE(arm, 'contacted') = 'contacted'
THEN CASE WHEN outcome = 'rebooked' THEN 1.0 ELSE 0.0 END END) AS DOUBLE)
AS contacted_rebook_rate,
CAST(AVG(CASE WHEN arm = 'holdout'
THEN CASE WHEN outcome = 'rebooked' THEN 1.0 ELSE 0.0 END END) AS DOUBLE)
AS holdout_rebook_rate,
CAST(SUM(CASE WHEN COALESCE(arm, 'contacted') = 'contacted'
THEN recovered_revenue ELSE 0 END) AS DOUBLE) AS recovered_revenue,
COUNT(DISTINCT run_id) AS runs
FROM {self.fq}.winback_outcomes
GROUP BY reason_category
ORDER BY contacted_rebook_rate DESC""")
if not rows:
return "No past outcomes yet. This is the first run, so there is no history to learn from."
for r in rows:
c, h = r["contacted_rebook_rate"], r["holdout_rebook_rate"]
r["lift"] = (c - h) if (c is not None and h is not None) else None
tc = sum(r["contacted"] for r in rows)
th = sum(r["held_out"] for r in rows)
rc = sum((r["contacted_rebook_rate"] or 0) * r["contacted"] for r in rows) / tc if tc else 0
rh = sum((r["holdout_rebook_rate"] or 0) * r["held_out"] for r in rows) / th if th else None
overall = (f"Overall: contacted rebook rate {rc:.1%} (n={tc}) vs holdout "
(f"{rh:.1%} (n={th}); lift {rc - rh:+.1%}." if rh is not None
else "n/a (no holdout patients yet)."))
return ("Past outreach results by reason category. 'lift' = contacted rebook rate minus holdout "
"rebook rate, i.e. what outreach actually caused. Holdout groups are small, so treat "
"per-category lift as a rough signal. Outcomes are simulated in this prototype.
"
_fmt(rows) + "
" + overall)
def get_overview(self):
d = self.lapsed_days
rows = self.sql(f"""
SELECT COUNT() AS patients,
SUM(CASE WHEN days_since_last_visit >= {d} THEN 1 ELSE 0 END) AS lapsed_patients,
CAST(ROUND(SUM(CASE WHEN days_since_last_visit >= {d} THEN est_annual_value ELSE 0 END)) AS DOUBLE)
AS lapsed_est_annual_value,
CAST(ROUND(AVG(days_since_last_visit)) AS DOUBLE) AS avg_days_since_visit,
MAX(as_of) AS data_as_of
FROM {self.fq}.patient_summary""")
buckets = self.sql(f"""
SELECT CASE WHEN days_since_last_visit < 90 THEN '1: <90 days'
WHEN days_since_last_visit < 180 THEN '2: 90-179 days'
WHEN days_since_last_visit < 365 THEN '3: 180-364 days'
ELSE '4: 365+ days' END AS since_last_visit,
COUNT() AS patients,
CAST(ROUND(SUM(est_annual_value)) AS DOUBLE) AS est_annual_value
FROM {self.fq}.patient_summary
GROUP BY 1 ORDER BY 1""")
return (f"Lapsed = no visit in {d}+ days.
" + _fmt(rows)
"
Patients by time since last visit:
" + _fmt(buckets))
def test_dropoff_driver(self, dimension):
if dimension not in DIMENSIONS:
return f"ERROR: dimension must be one of {list(DIMENSIONS)}"
d = self.lapsed_days
rows = self.sql(f"""
SELECT {DIMENSIONS[dimension]} AS grp,
COUNT() AS n,
CAST(AVG(CASE WHEN days_since_last_visit >= {d} THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS lapse_rate
FROM {self.fq}.patient_summary
GROUP BY 1
HAVING COUNT() >= 30
ORDER BY lapse_rate DESC""")
if len(rows) < 2:
return f"Not enough groups to compare for {dimension}."
total = sum(r["n"] for r in rows)
p = sum(r["n"] * r["lapse_rate"] for r in rows) / total
hi, lo = rows[0], rows[-1]
se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / hi["n"] + 1 / lo["n"]))
z = (hi["lapse_rate"] - lo["lapse_rate"]) / se
spread = hi["lapse_rate"] - lo["lapse_rate"]
notable = abs(z) >= 3 and spread >= 0.05
verdict = ("NOTABLE: this dimension meaningfully affects drop-off."
if notable else
"NOT NOTABLE: differences are small and consistent with random noise.")
return (f"Drop-off by {dimension} (overall lapse rate {p:.1%}):
{_fmt(rows, 25)}
"
f"Highest {hi['grp']} {hi['lapse_rate']:.1%} vs lowest {lo['grp']} {lo['lapse_rate']:.1%}; "
f"spread {spread:.1%}, z={z:.1f}. {verdict}")
def get_location_capacity(self):
rows = self.sql(f"""
SELECT location_id, location_name, region, capacity_patients_per_day,
avg_visits_per_day, utilization_pct, open_slots_per_week, active_providers
FROM {self.fq}.location_capacity
ORDER BY utilization_pct""")
for r in rows:
r["remaining_budget_this_run"] = self._remaining(r["location_id"])
return ("Clinics from least to most utilized. remaining_budget_this_run = how many more "
"patients you may route there in this run.
" + _fmt(rows))
def find_lapsed_patients(self, situation=None, home_location_id=None, limit=15):
d = self.lapsed_days
limit = max(1, min(int(limit), 30))
where = [f"days_since_last_visit >= {d}",
f"""patient_id NOT IN (SELECT patient_id FROM {self.fq}.winback_actions
WHERE action_type IN ('contact', 'holdout') OR run_id = {_lit(self.run_id)})"""]
if situation:
if situation not in FILTERS:
return f"ERROR: situation must be one of {list(FILTERS)} or omitted"
where.append(FILTERS[situation])
if home_location_id:
if not SAFE_ID.match(home_location_id):
return "ERROR: invalid home_location_id"
where.append(f"home_location_id = {_lit(home_location_id)}")
rows = self.sql(f"""
SELECT patient_id, days_since_last_visit, n_visits, est_annual_value,
home_location_id, last_location_id, last_provider_active,
last_service_type, consult_only, n_no_shows, n_cancellations, n_appointments
FROM {self.fq}.patient_summary
WHERE {' AND '.join(where)}
ORDER BY est_annual_value DESC
LIMIT {limit}""")
label = situation or "any"
return (f"Lapsed patients not yet contacted (situation={label}), highest value first:
"
_fmt(rows))
def get_patient_profile(self, patient_id):
if not SAFE_ID.match(str(patient_id)):
return "ERROR: invalid patient_id"
if self.profile_calls >= 3:
return ("ERROR: profile limit of 3 per run reached. "
"Decide using the facts find_lapsed_patients already returned.")
self.profile_calls += 1
pid = _lit(patient_id)
summary = self.sql(f"SELECT * FROM {self.fq}.patient_summary WHERE patient_id = {pid}")
if not summary:
return f"ERROR: patient {patient_id} not found"
visits = self.sql(f"""
SELECT visit_date, service_type, location_id, provider_id, revenue
FROM {self.fq}.visits WHERE patient_id = {pid}
ORDER BY visit_date DESC LIMIT 8""")
appts = self.sql(f"""
SELECT status, COUNT() AS n FROM {self.fq}.appointments
WHERE patient_id = {pid} GROUP BY status""")
home = summary[0]["home_location_id"]
return (f"Patient {patient_id} summary:
{_fmt(summary)}
Recent visits:
{_fmt(visits)}"
f"
Appointment outcomes:
{_fmt(appts)}"
f"
Home clinic {home} remaining budget this run: {self._remaining(home)}")
def list_providers(self, location_id):
if not SAFE_ID.match(str(location_id)):
return "ERROR: invalid location_id"
rows = self.sql(f"""
SELECT pr.provider_id, pr.specialty, pr.employment_type, pr.hire_date,
COUNT(v.visit_id) AS visits_handled
FROM {self.fq}.providers pr
LEFT JOIN {self.fq}.visits v ON v.provider_id = pr.provider_id
WHERE pr.location_id = {_lit(location_id)} AND pr.active_flag
GROUP BY pr.provider_id, pr.specialty, pr.employment_type, pr.hire_date
ORDER BY visits_handled""")
return f"Active providers at {location_id}:
" + _fmt(rows)
def record_winback_action(self, patient_id, target_location_id, reason_category,
reasoning, outreach_message, priority, target_provider_id=None):
if self.contacts >= self.max_actions:
return f"ERROR: outreach limit of {self.max_actions} contacts reached. Stop and summarize."
if not SAFE_ID.match(str(patient_id)) or not SAFE_ID.match(str(target_location_id)):
return "ERROR: invalid patient_id or target_location_id"
if reason_category not in REASONS:
return f"ERROR: reason_category must be one of {REASONS}"
if priority not in ("high", "medium", "low"):
return "ERROR: priority must be high, medium or low"
if target_location_id not in self._budgets():
return f"ERROR: unknown location {target_location_id}"
if self._remaining(target_location_id) <= 0:
return (f"ERROR: {target_location_id} has no remaining capacity budget this run. "
"Choose another clinic (ideally in the same region) or skip.")
pid = _lit(patient_id)
found = self.sql(f"SELECT est_annual_value FROM {self.fq}.patient_summary WHERE patient_id = {pid}")
if not found:
return f"ERROR: patient {patient_id} not found"
dup = self.sql(f"""SELECT 1 AS x FROM {self.fq}.winback_actions
WHERE patient_id = {pid} AND (action_type IN ('contact', 'holdout') OR run_id = {_lit(self.run_id)})
LIMIT 1""")
if dup:
return f"ERROR: patient {patient_id} was already handled. Pick someone else."
if target_provider_id:
if not SAFE_ID.match(str(target_provider_id)):
return "ERROR: invalid target_provider_id"
prov = self.sql(f"""SELECT location_id, active_flag FROM {self.fq}.providers
WHERE provider_id = {_lit(target_provider_id)}""")
if not prov or not prov[0]["active_flag"] or prov[0]["location_id"] != target_location_id:
return f"ERROR: {target_provider_id} is not an active provider at {target_location_id}"
value = float(found[0]["est_annual_value"] or 0)
if self._in_holdout(patient_id):
self.sql(f"""
INSERT INTO {self.fq}.winback_actions
(run_id, created_at, action_type, patient_id, target_location_id, target_provider_id,
reason_category, reasoning, outreach_message, priority, est_annual_value, status, reviewer_note)
VALUES ({_lit(self.run_id)}, current_timestamp(), 'holdout', {pid}, {_lit(target_location_id)},
{_lit(target_provider_id)}, {_lit(reason_category)}, {_lit(reasoning)},
{_lit(outreach_message)}, {_lit(priority)}, {value}, 'holdout', NULL)""")
self.holdouts += 1
return (f"{patient_id} was randomly assigned to the HOLDOUT group: recorded but NOT contacted, "
f"so we can measure how many return on their own. Contact budget unchanged "
f"({self.contacts}/{self.max_actions}). Do not choose this patient again; pick the next candidate.")
self.sql(f"""
INSERT INTO {self.fq}.winback_actions
(run_id, created_at, action_type, patient_id, target_location_id, target_provider_id,
reason_category, reasoning, outreach_message, priority, est_annual_value, status, reviewer_note)
VALUES ({_lit(self.run_id)}, current_timestamp(), 'contact', {pid}, {_lit(target_location_id)},
{_lit(target_provider_id)}, {_lit(reason_category)}, {_lit(reasoning)},
{_lit(outreach_message)}, {_lit(priority)}, {value}, 'proposed', NULL)""")
self.used[target_location_id] = self.used.get(target_location_id, 0) + 1
self.contacts += 1
return (f"Recorded contact for {patient_id} -> {target_location_id} "
f"(est. annual value ${value:,.0f}). Contacts so far: {self.contacts}/{self.max_actions}.")
def skip_patient(self, patient_id, reasoning):
if not SAFE_ID.match(str(patient_id)):
return "ERROR: invalid patient_id"
pid = _lit(patient_id)
found = self.sql(f"SELECT est_annual_value FROM {self.fq}.patient_summary WHERE patient_id = {pid}")
if not found:
return f"ERROR: patient {patient_id} not found"
self.sql(f"""
INSERT INTO {self.fq}.winback_actions
(run_id, created_at, action_type, patient_id, target_location_id, target_provider_id,
reason_category, reasoning, outreach_message, priority, est_annual_value, status, reviewer_note)
VALUES ({_lit(self.run_id)}, current_timestamp(), 'skip', {pid}, NULL, NULL, NULL,
{_lit(reasoning)}, NULL, NULL, {float(found[0]['est_annual_value'] or 0)}, 'skipped', NULL)""")
self.skips += 1
return f"Skipped {patient_id}. Skips so far: {self.skips}."
# ---------- LLM wiring ----------
def call(self, name, arguments_json):
fn = getattr(self, name, None)
if name not in TOOL_NAMES or fn is None:
return f"ERROR: unknown tool {name}"
try:
args = json.loads(arguments_json or "{}")
except json.JSONDecodeError:
return "ERROR: arguments were not valid JSON"
try:
return fn(args)
except TypeError as e:
return f"ERROR: bad arguments for {name}: {e}"
except Exception as e: # surface SQL or other errors to the agent
return f"ERROR in {name}: {str(e)[:400]}"

def _fn(name, description, properties=None, required=None):
return {"type": "function", "function": {
"name": name, "description": description,
"parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}

TOOL_SPECS = [
_fn("review_past_outcomes",
"See how past outreach performed by reason category, including lift vs. the holdout group. Call this first."),
_fn("get_overview",
"Count lapsed patients and their estimated annual value, and how long patients have been away."),
_fn("test_dropoff_driver",
"Test whether one dimension explains which patients lapse. Returns lapse rate per group and a verdict.",
{"dimension": {"type": "string", "enum": list(DIMENSIONS)}}, ["dimension"]),
_fn("get_location_capacity",
"List clinics with utilization, open slots, region, and how many more patients you may route there this run."),
_fn("find_lapsed_patients",
"Find lapsed patients not yet contacted, highest estimated value first. Optionally filter by situation or home clinic.",
{"situation": {"type": "string", "enum": list(FILTERS)},
"home_location_id": {"type": "string"},
"limit": {"type": "integer", "description": "1-30, default 15"}}),
_fn("get_patient_profile",
"Full facts for one patient: summary, recent visits, appointment outcomes, home clinic budget.",
{"patient_id": {"type": "string"}}, ["patient_id"]),
_fn("list_providers",
"List active providers at a clinic (least busy first). Use when a patient's provider is inactive.",
{"location_id": {"type": "string"}}, ["location_id"]),
_fn("record_winback_action",
"Record a decision to contact a lapsed patient, routed to a clinic, for staff approval. "
"Some patients are automatically assigned to a holdout group and not contacted.",
{"patient_id": {"type": "string"},
"target_location_id": {"type": "string"},
"target_provider_id": {"type": "string", "description": "Optional active provider at the target clinic"},
"reason_category": {"type": "string", "enum": REASONS},
"reasoning": {"type": "string", "description": "1-2 sentences citing the specific facts behind this decision"},
"outreach_message": {"type": "string", "description": "Warm message, max 3 sentences, no medical claims"},
"priority": {"type": "string", "enum": ["high", "medium", "low"]}},
["patient_id", "target_location_id", "reason_category", "reasoning", "outreach_message", "priority"]),
_fn("skip_patient",
"Record a decision NOT to contact a patient, with the reason.",
{"patient_id": {"type": "string"}, "reasoning": {"type": "string"}},
["patient_id", "reasoning"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SPECS}
