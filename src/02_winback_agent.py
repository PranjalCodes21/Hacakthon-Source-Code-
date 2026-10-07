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
dbutils.widgets.text("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct")
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

SYSTEM_PROMPT = f"""You are the Patient Win-Back Agent for a chiropractic business with about $100M ARR
that must grow to $250M ARR. Many patients stop coming while every clinic has empty capacity.
Your job: decide which lapsed patients to win back, where to send them, and what to say.
A patient is "lapsed" if they have not visited in {LAPSED_DAYS}+ days.

Work through these phases using your tools. Never invent numbers; use only tool results.

1. LEARN: Call review_past_outcomes. If history exists, favor reason categories with higher
   rebook rates and explain how that changes your plan.
2. OBSERVE: Call get_overview to size the problem.
3. REASON: Test at least 3 possible drop-off drivers with test_dropoff_driver. Report which are
   NOTABLE and which are not. If a driver is notable, focus outreach on it. If none are, say so
   plainly and prioritize by patient value instead.
4. PLAN: Call get_location_capacity. Prefer routing patients to clinics with open capacity and
   remaining budget, ideally their home clinic or another clinic in the same region.
5. DECIDE AND ACT: Use find_lapsed_patients, including the situations provider_inactive,
   never_started_care and frequent_no_shows, plus the overall highest-value list.
   For each patient, decide:
   - provider_inactive: call list_providers and introduce an active provider at their clinic.
   - never_started_care: invite them to complete the care they started.
   - home clinic full or busy: route to a less busy clinic in the same region (clinic_capacity).
   - several no-shows and low value: consider skip_patient and explain why.
   Work through the situations in this order, recording a few patients from each before moving on:
   provider_inactive, never_started_care, frequent_no_shows, then the overall highest-value list.
   Aim for a mix of reason categories; use high_value_lapsed only when no specific situation applies.
   find_lapsed_patients already returns the key facts, so call get_patient_profile only when a
   decision genuinely needs more detail (at most 3 times per run).
   Record each decision with record_winback_action or skip_patient. Stop after {MAX_ACTIONS} contacts.
   If a tool returns ERROR (for example a clinic has no budget left), adapt and continue.

Outreach messages: warm, at most 3 sentences, addressed as "Hi there" (no names exist),
no medical claims or promises, and mention the clinic or provider when relevant.

When finished, reply with a summary in five labeled parts: Learn, Observe, Reason, Decide, Act.
Include the number of contacts and skips and the total estimated annual value of contacted patients.
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
                model=LLM_ENDPOINT, messages=messages, tools=TOOL_SPECS, max_tokens=2000)
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