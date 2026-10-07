# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Simulate outreach outcomes (Measure step)
# MAGIC The data is synthetic, so we can't observe real patient responses. This notebook simulates
# MAGIC them for the latest run using **stated assumptions** below. The next agent run reads these
# MAGIC results with `review_past_outcomes` and adjusts its strategy. With real clinic data, this
# MAGIC notebook would be replaced by actual rebooking data.

# COMMAND ----------

import random
from datetime import date

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "chiro_hackathon")
FQ = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"

# ASSUMPTIONS: chance a contacted patient rebooks, by why they were contacted.
BASE_REBOOK_RATE = {
    "provider_inactive": 0.30,      # a fresh provider introduction removes the reason they left
    "never_started_care": 0.20,
    "clinic_capacity": 0.25,        # easier scheduling at a less busy clinic
    "high_value_lapsed": 0.18,
    "frequent_no_shows": 0.10,
    "general_reengagement": 0.12,
}
# Patients who have been away longer are harder to win back.
def recency_factor(days):
    return max(0.4, 1 - days / 1000)

REVENUE_SHARE_IF_REBOOKED = 0.5    # assume half of their historical annual value comes back

# COMMAND ----------

contacts = spark.sql(f"""
    SELECT a.run_id, a.patient_id, a.reason_category, a.target_location_id, a.est_annual_value,
           p.days_since_last_visit
    FROM {FQ}.winback_actions a
    JOIN {FQ}.patient_summary p ON a.patient_id = p.patient_id
    WHERE a.action_type = 'contact'
      AND a.status IN ('proposed', 'approved')
      AND a.run_id NOT IN (SELECT DISTINCT run_id FROM {FQ}.winback_outcomes)""").collect()
if not contacts:
    dbutils.notebook.exit("No new contacts to simulate.")

rng = random.Random(42)  # repeatable
rows = []
for c in contacts:
    prob = BASE_REBOOK_RATE.get(c["reason_category"], 0.12) * recency_factor(c["days_since_last_visit"])
    r = rng.random()
    if r < prob:
        outcome, revenue = "rebooked", round(float(c["est_annual_value"]) * REVENUE_SHARE_IF_REBOOKED, 2)
    elif r < prob + 0.15:
        outcome, revenue = "declined", 0.0
    else:
        outcome, revenue = "no_response", 0.0
    rows.append((c["run_id"], c["patient_id"], c["reason_category"], c["target_location_id"],
                 outcome, revenue, True, date.today()))

spark.createDataFrame(
    rows,
    "run_id STRING, patient_id STRING, reason_category STRING, target_location_id STRING, "
    "outcome STRING, recovered_revenue DOUBLE, simulated BOOLEAN, outcome_date DATE",
).write.mode("append").saveAsTable(f"{FQ}.winback_outcomes")

print(f"Simulated {len(rows)} outcomes")

# COMMAND ----------

display(spark.sql(f"""
    SELECT reason_category,
           COUNT(*) AS contacted,
           SUM(CASE WHEN outcome = 'rebooked' THEN 1 ELSE 0 END) AS rebooked,
           ROUND(SUM(recovered_revenue)) AS recovered_revenue
    FROM {FQ}.winback_outcomes
    GROUP BY reason_category ORDER BY recovered_revenue DESC"""))