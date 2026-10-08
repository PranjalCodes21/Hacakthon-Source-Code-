# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Patient Win-Back Agent
# MAGIC **Learn → Observe → Reason → Decide → Act.** The LLM runs on a Databricks model serving
# MAGIC endpoint and works through the tools in `winback_tools.py`. Decisions go to
# MAGIC `winback_actions` for staff approval; each run is logged to `winback_runs`.

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-sdk openai mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import uuid

sys.path.append(os.getcwd())  # so winback_tools.py next to this notebook can be imported
from databricks.sdk import WorkspaceClient
from winback_tools import WinbackTools, TOOL_SPECS

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "chiro_hackathon")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-oss-120b")
dbutils.widgets.text("max_actions", "15")
dbutils.widgets.text("lapsed_days", "180")
dbutils.widgets.text("capacity_share", "0.02")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
MAX_ACTIONS = int(dbutils.widgets.get("max_actions"))
LAPSED_DAYS = int(dbutils.widgets.get("lapsed_days"))
CAPACITY_SHARE = float(dbutils.widgets.get("capacity_share"))
FQ = f"{CATALOG}.{SCHEMA}"

RUN_ID = str(uuid.uuid4())
MAX_STEPS = 60

tools = WinbackTools(
    run_sql=lambda q: [r.asDict() for r in spark.sql(q).collect()],
    catalog=CATALOG, schema=SCHEMA, run_id=RUN_ID,
    lapsed_days=LAPSED_DAYS, max_actions=MAX_ACTIONS, capacity_share=CAPACITY_SHARE,
)
print(f"Run {RUN_ID} | data {FQ} | model {LLM_ENDPOINT} | up to {MAX_ACTIONS} contacts")

# COMMAND ----------

# MLflow tracing records every model call and tool call (great for the demo). Never blocks the run.
try:
    import mlflow
    mlflow.openai.autolog()
except Exception as e:
    print(f"MLflow tracing not enabled: {e}")

# COMMAND ----------

SYSTEM_PROMPT = f"""You are the Patient Win-Back Agent for a chiropractic business at about $100M ARR
that must grow to $250M ARR.
THE PROBLEM YOU SOLVE
A patient is "lapsed" if they have not visited in {LAPSED_DAYS}+ days. There are tens of thousands of them.
Drafting messages is cheap, but every patient you contact becomes work for a human: front-desk staff must
follow up, and a clinic must have a real slot with an active provider. Staff can handle only {MAX_ACTIONS}
win-back contacts this run. Your job: decide which lapsed patients deserve those contacts, where to send
each one, and what to say, so limited follow-up time recovers as much revenue as possible.
Never invent numbers; use only tool results. If a tool returns ERROR, read it, adapt, and continue.
PHASES
LEARN: Call review_past_outcomes. If history exists, compare contacted patients with the holdout group
(patients chosen but deliberately not contacted). The difference is the lift your outreach actually
Shift effort toward situations with higher lift, and state in one or two sentences how this
run's plan differs from the last because of it. If there is no history, say this is the first run.
OBSERVE: Call get_overview to size the lapsed pool and its value.
REASON: Test at least 3 possible drop-off drivers with test_dropoff_driver and report which are NOTABLE
and which are not. If a driver is notable, weight outreach toward patients it affects. If none are,
say so plainly: there is no single cause to fix, so the right move is to spend the limited contacts
where expected return is highest.
PLAN: Call get_location_capacity. Note which clinics have open capacity and remaining budget. Prefer a
patient's home clinic, or another clinic in the same region if the home clinic has no budget left.
DECIDE AND ACT:
Build a candidate pool by calling find_lapsed_patients for each situation (provider_inactive,
never_started_care, frequent_no_shows) and for the overall highest-value list. Then choose across the
whole pool, not list by list.
For each candidate, judge expected return = estimated annual value x likelihood of winning them back.
Judge likelihood only from facts the tools return:
fewer days since last visit: more winnable
more completed visits before lapsing: an established habit, more winnable
consult only: never started care; a concrete invitation to start can work, but commitment is unproven
several no-shows or cancellations: less winnable, and they may waste a slot
last provider no longer active: the relationship is gone; winnable only if you introduce a specific
active provider at their clinic (use list_providers)
A high-value patient who is unlikely to return can rank below a moderate-value patient who is easy to
win back. Contact the candidates with the highest expected return first.
Use skip_patient when expected return is too low to justify a contact (for example, low value plus
frequent no-shows), and explain the trade-off in one sentence. Skipping is a good decision, not a failure.
For each contact, call record_winback_action with: the reason category (the most specific situation that
applies; use high_value_lapsed only if none applies), the target clinic and provider, one or two
sentences of reasoning that cite the actual numbers (days away, completed visits, no-shows and
cancellations, value) and the trade-off you weighed, and the outreach message. Write each patient's
reasoning and message fresh; never reuse wording across patients. The reason category must match
the facts: use frequent_no_shows only for 2+ no-shows, provider_inactive only when the last provider
is inactive.
record_winback_action automatically assigns some patients to the holdout group. That is expected:
holdout patients do not use the contact budget, and you must not try to contact them again.
find_lapsed_patients already returns the key facts; call get_patient_profile only when a decision
genuinely needs more detail (at most 3 times per run).
Stop after {MAX_ACTIONS} contacts. If you stop early, explain why.
OUTREACH MESSAGES
Warm, at most 3 sentences, start with "Hi there" (no names exist), no medical claims or promises,
and mention the clinic by its name (never an ID like LOC011). Never mention money, value estimates,
or internal IDs (like PRV0044) in a message. Refer to a provider only by role from list_providers
(for example "one of our chiropractors"), and use "Dr." only for chiropractors.
FINAL SUMMARY
Reply with five labeled parts: Learn, Observe, Reason, Decide, Act. Include the number of contacts,
skips, and holdout patients; the total estimated annual value of contacted patients; and one sentence on
how this run's priorities differed from the last run (or that it was the first run).
End after the summary; do not ask the user follow-up questions."""

# COMMAND ----------

client = WorkspaceClient().serving_endpoints.get_open_ai_client()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Run this week's patient win-back review."},
]

import time
from openai import RateLimitError

def ask_model(messages, max_retries=6):
    """Call the model, waiting and retrying if Free Edition's rate limit is hit."""
    wait = 10
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=LLM_ENDPOINT, messages=messages, tools=TOOL_SPECS, max_tokens=4000)
        except RateLimitError:
            print(f"    (rate limited, waiting {wait}s and retrying...)")
            time.sleep(wait)
            wait = min(wait * 2, 60)
    raise RuntimeError("Still rate limited after several retries. Try again in a few minutes.")

final_answer = None
for step in range(1, MAX_STEPS + 1):
    time.sleep(2)  # short pause between calls to stay under Free Edition limits
    resp = ask_model(messages)
    msg = resp.choices[0].message

    if not msg.tool_calls:
        final_answer = msg.content
        break

    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [{"id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                       for tc in msg.tool_calls],
    })
    for tc in msg.tool_calls:
        result = tools.call(tc.function.name, tc.function.arguments)
        print(f"[step {step}] {tc.function.name}({(tc.function.arguments or '')[:150]})\n    -> {str(result)[:300]}\n")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
else:
    final_answer = "Stopped: reached the step limit before the agent finished."

print("\n===== AGENT SUMMARY =====\n")
print(final_answer)

# COMMAND ----------

# Log the run.
totals = spark.sql(f"""
    SELECT COALESCE(SUM(CASE WHEN action_type = 'contact' THEN est_annual_value END), 0) AS v
    FROM {FQ}.winback_actions WHERE run_id = '{RUN_ID}'""").collect()[0]["v"]

spark.createDataFrame(
    [(RUN_ID, LLM_ENDPOINT, tools.contacts, tools.skips, float(totals), final_answer or "")],
    "run_id STRING, llm_endpoint STRING, n_contacts INT, n_skips INT, total_est_annual_value DOUBLE, summary STRING",
).selectExpr("run_id", "current_timestamp() AS created_at", "llm_endpoint", "n_contacts", "n_skips",
             "total_est_annual_value", "summary") \
 .write.mode("append").saveAsTable(f"{FQ}.winback_runs")

display(spark.sql(f"""
    SELECT action_type, patient_id, target_location_id, target_provider_id, reason_category,
           priority, est_annual_value, reasoning, outreach_message
    FROM {FQ}.winback_actions WHERE run_id = '{RUN_ID}'
    ORDER BY action_type, est_annual_value DESC"""))