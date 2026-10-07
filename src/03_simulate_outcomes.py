Databricks notebook source
/// script
[tool.databricks.environment]
environment_version = "6"
///
MAGIC %md
MAGIC # Simulate outreach outcomes (Measure step)
MAGIC The data is synthetic, so we can't observe real patient responses. This notebook simulates
MAGIC them for every run that doesn't have outcomes yet, using stated assumptions below.
MAGIC
MAGIC Two groups are simulated:
MAGIC - contacted: patients the agent chose and staff would reach out to
MAGIC - holdout: patients the agent chose but who were randomly held back and NOT contacted
MAGIC
MAGIC The difference in rebook rate between the two is the lift outreach caused.
MAGIC
MAGIC The chance of rebooking depends on patient facts (time away, visit habit, show rate),
MAGIC NOT on the reason category the agent picked. So results reward the agent for judging
MAGIC winnability well, instead of just echoing rates we typed in per category.
MAGIC With real clinic data, this notebook would be replaced by actual rebookings; the holdout
MAGIC comparison and the agent's learning loop would stay the same.
COMMAND ----------
import random
from datetime import date
dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "chiro_hackathon")
FQ = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
---- ASSUMPTIONS (state these openly in the pitch) ----
REVENUE_SHARE_IF_REBOOKED = 0.5 # half of a patient's historical annual value comes back

def _facts(c):
"""Turn patient facts into 0-1 signals."""
days = c["days_since_last_visit"] or 0
recency = max(0.0, 1 - days / 1000) # recently lapsed -> closer to 1
habit = min(c["n_visits"] or 0, 10) / 10 # established routine -> closer to 1
appts = max(c["n_appointments"] or 0, 1)
missed = (c["n_no_shows"] or 0) + (c["n_cancellations"] or 0)
show_rate = max(0.0, 1 - missed / appts) # reliable patient -> closer to 1
return recency, habit, show_rate

def win_prob(c):
"""ASSUMPTION: chance a CONTACTED patient rebooks, from their facts."""
recency, habit, show_rate = _facts(c)
p = (0.05 + 0.20 * recency + 0.10 * habit) * show_rate
return min(max(p, 0.02), 0.45)

def natural_prob(c):
"""ASSUMPTION: chance a HOLDOUT patient returns on their own, with no outreach."""
recency, habit, show_rate = _facts(c)
p = (0.02 + 0.04 * recency + 0.02 * habit) * show_rate
return min(max(p, 0.01), 0.10)
COMMAND ----------
decisions = spark.sql(f"""
SELECT a.run_id, a.patient_id, a.action_type, a.reason_category, a.target_location_id,
a.est_annual_value, p.days_since_last_visit, p.n_visits,
p.n_no_shows, p.n_cancellations, p.n_appointments
FROM {FQ}.winback_actions a
JOIN {FQ}.patient_summary p ON a.patient_id = p.patient_id
WHERE ((a.action_type = 'contact' AND a.status IN ('proposed', 'approved'))
OR a.action_type = 'holdout')
AND a.run_id NOT IN (SELECT DISTINCT run_id FROM {FQ}.winback_outcomes)""").collect()
if not decisions:
dbutils.notebook.exit("No new decisions to simulate.")
rng = random.Random(42) # repeatable
rows = []
for c in decisions:
if c["action_type"] == "holdout":
arm, prob = "holdout", natural_prob(c)
else:
arm, prob = "contacted", win_prob(c)
r = rng.random()
if r < prob:
outcome, revenue = "rebooked", round(float(c["est_annual_value"] or 0) * REVENUE_SHARE_IF_REBOOKED, 2)
elif arm == "contacted" and r < prob + 0.15:
outcome, revenue = "declined", 0.0 # only contacted patients can decline
else:
outcome, revenue = "no_response", 0.0
rows.append((c["run_id"], c["patient_id"], c["reason_category"], c["target_location_id"],
outcome, revenue, True, date.today(), arm))
(spark.createDataFrame(
rows,
"run_id STRING, patient_id STRING, reason_category STRING, target_location_id STRING, "
"outcome STRING, recovered_revenue DOUBLE, simulated BOOLEAN, outcome_date DATE, arm STRING",
)
.write.mode("append")
.option("mergeSchema", "true")
.saveAsTable(f"{FQ}.winback_outcomes"))
n_hold = sum(1 for r in rows if r[-1] == "holdout")
print(f"Simulated {len(rows)} outcomes ({len(rows) - n_hold} contacted, {n_hold} holdout)")
COMMAND ----------
Contacted vs. holdout, by reason category. Older rows without an arm count as contacted.
display(spark.sql(f"""
SELECT reason_category,
COALESCE(arm, 'contacted') AS arm,
COUNT() AS patients,
SUM(CASE WHEN outcome = 'rebooked' THEN 1 ELSE 0 END) AS rebooked,
ROUND(AVG(CASE WHEN outcome = 'rebooked' THEN 1.0 ELSE 0.0 END), 3) AS rebook_rate,
ROUND(SUM(recovered_revenue)) AS revenue
FROM {FQ}.winback_outcomes
GROUP BY 1, 2 ORDER BY 1, 2"""))
COMMAND ----------
Overall lift: what outreach caused beyond patients who would have returned anyway.
display(spark.sql(f"""
SELECT COALESCE(arm, 'contacted') AS arm,
COUNT() AS patients,
ROUND(AVG(CASE WHEN outcome = 'rebooked' THEN 1.0 ELSE 0.0 END), 3) AS rebook_rate
FROM {FQ}.winback_outcomes
GROUP BY 1 ORDER BY 1"""))
