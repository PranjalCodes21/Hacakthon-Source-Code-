# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Setup tables for the Patient Win-Back Agent
# MAGIC Builds two summary tables from the generator's raw data (so the agent's tools are fast
# MAGIC and simple), plus the tables the agent writes to. Safe to re-run.
# MAGIC
# MAGIC Note: the generator's `patients.status` and `patients.churn_risk_score` columns are random,
# MAGIC so we derive everything from the real `visits` and `appointments` tables instead.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "chiro_hackathon")
FQ = f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
print(f"Using {FQ}")

# COMMAND ----------

# One row per patient with the facts the agent reasons about.
# "as_of" = the latest visit date in the data (synthetic dates may not line up with today).
spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.patient_summary AS
WITH ref AS (
  SELECT MAX(visit_date) AS as_of FROM {FQ}.visits
),
v AS (
  SELECT patient_id,
         COUNT(*)                               AS n_visits,
         SUM(revenue)                           AS total_revenue,
         MIN(visit_date)                        AS first_visit,
         MAX(visit_date)                        AS last_visit,
         MAX_BY(location_id, visit_date)        AS last_location_id,
         MAX_BY(provider_id, visit_date)        AS last_provider_id,
         MAX_BY(service_type, visit_date)       AS last_service_type,
         SUM(CASE WHEN service_type = 'Initial Consultation' THEN 1 ELSE 0 END) AS n_consults
  FROM {FQ}.visits
  GROUP BY patient_id
),
a AS (
  SELECT patient_id,
         COUNT(*)                                                AS n_appointments,
         SUM(CASE WHEN status = 'No-Show'   THEN 1 ELSE 0 END)   AS n_no_shows,
         SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END)   AS n_cancellations
  FROM {FQ}.appointments
  GROUP BY patient_id
)
SELECT p.patient_id,
       p.home_location_id,
       p.acquisition_source,
       p.age_band,
       v.n_visits,
       CAST(ROUND(v.total_revenue, 2) AS DOUBLE)              AS total_revenue,
       CAST(ROUND(v.total_revenue / v.n_visits, 2) AS DOUBLE) AS avg_revenue_per_visit,
       v.first_visit,
       v.last_visit,
       DATEDIFF(ref.as_of, v.last_visit)                      AS days_since_last_visit,
       CAST(ROUND(v.total_revenue /
            GREATEST(DATEDIFF(v.last_visit, v.first_visit) / 365.0, 1.0), 2) AS DOUBLE) AS est_annual_value,
       v.last_location_id,
       v.last_provider_id,
       COALESCE(pr.active_flag, false)                        AS last_provider_active,
       v.last_service_type,
       (v.n_consults = v.n_visits)                            AS consult_only,
       COALESCE(a.n_appointments, 0)                          AS n_appointments,
       COALESCE(a.n_no_shows, 0)                              AS n_no_shows,
       COALESCE(a.n_cancellations, 0)                         AS n_cancellations,
       ref.as_of
FROM {FQ}.patients p
JOIN v            ON p.patient_id = v.patient_id
LEFT JOIN a       ON p.patient_id = a.patient_id
LEFT JOIN {FQ}.providers pr ON v.last_provider_id = pr.provider_id
CROSS JOIN ref
""")

# COMMAND ----------

# One row per clinic: how busy it is and how much room it has.
spark.sql(f"""
CREATE OR REPLACE TABLE {FQ}.location_capacity AS
WITH d AS (
  SELECT location_id, COUNT(*) AS n_visits, COUNT(DISTINCT visit_date) AS n_days
  FROM {FQ}.visits
  GROUP BY location_id
),
pr AS (
  SELECT location_id, COUNT(*) AS active_providers
  FROM {FQ}.providers
  WHERE active_flag
  GROUP BY location_id
)
SELECT l.location_id,
       l.location_name,
       l.region,
       l.capacity_patients_per_day,
       CAST(ROUND(d.n_visits / d.n_days, 1) AS DOUBLE)                                   AS avg_visits_per_day,
       CAST(ROUND(100 * d.n_visits / d.n_days / l.capacity_patients_per_day, 1) AS DOUBLE) AS utilization_pct,
       CAST(FLOOR((l.capacity_patients_per_day - d.n_visits / d.n_days) * 5) AS INT)     AS open_slots_per_week,
       COALESCE(pr.active_providers, 0)                                                  AS active_providers
FROM {FQ}.locations l
JOIN d       ON l.location_id = d.location_id
LEFT JOIN pr ON l.location_id = pr.location_id
""")

# COMMAND ----------

# What the agent decides (Act step). action_type = 'contact' or 'skip'.
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.winback_actions (
  run_id STRING,
  created_at TIMESTAMP,
  action_type STRING,
  patient_id STRING,
  target_location_id STRING,
  target_provider_id STRING,
  reason_category STRING,
  reasoning STRING,
  outreach_message STRING,
  priority STRING,
  est_annual_value DOUBLE,
  status STRING,          -- proposed / approved / rejected
  reviewer_note STRING
)
""")

# Results of past outreach (Measure step). In this prototype outcomes are simulated.
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.winback_outcomes (
  run_id STRING,
  patient_id STRING,
  reason_category STRING,
  target_location_id STRING,
  outcome STRING,         -- rebooked / no_response / declined
  recovered_revenue DOUBLE,
  simulated BOOLEAN,
  outcome_date DATE
)
""")

# One row per agent run, for the app's history view.
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {FQ}.winback_runs (
  run_id STRING,
  created_at TIMESTAMP,
  llm_endpoint STRING,
  n_contacts INT,
  n_skips INT,
  total_est_annual_value DOUBLE,
  summary STRING
)
""")

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS patients, ROUND(AVG(days_since_last_visit)) AS avg_days_since_visit, MAX(as_of) AS as_of FROM {FQ}.patient_summary"))
display(spark.sql(f"SELECT * FROM {FQ}.location_capacity ORDER BY utilization_pct"))