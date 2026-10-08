# Hackathon-Source-Code-
# Patient Win-Back Agent

An agentic AI prototype on Databricks that decides which lapsed patients a chiropractic business should win back, where to send them, and what to say, then measures whether it worked.

Built for the Xorbix × ACM Databricks Hackathon at the University of Iowa, October 2026.

## The problem

A chiropractic group at ~$100M ARR wants to reach $250M. In the synthetic data, **34,334 of 59,243 patients (58%) haven't visited in 180+ days**, worth about **$6.7M a year**, while every clinic runs at 10–32% capacity.

An LLM can draft 34,000 win-back messages for pennies, but staff can only follow up with a handful each week. So the real question is: **"Your team can work N patients this week. Which N?"**

## How it works

Each run, the agent:

1. **Learns** from past outcomes, comparing contacted patients against a randomized holdout group.
2. **Observes** the size and value of the lapsed pool.
3. **Reasons** by testing possible drop-off drivers (clinic, age band, acquisition source, service type, provider status) with spread and z-score checks. On this data, none is notable, so there's no single cause to fix.
4. **Decides** using expected return = patient value × likelihood of winning them back, within an outreach cap and per-clinic capacity. It can skip patients, with a reason.
5. **Acts** by writing each decision (clinic, provider, reasoning, drafted message) to a Delta table for staff approval.

A separate step simulates outcomes, and the next run learns from them.

**Why it's an agent, not a dashboard:** it chooses what to investigate, makes trade-offs between value and winnability, can say no, re-plans when tools reject an action, and changes decisions based on past results. Every step is traced with MLflow.

## Stack

- **Databricks Free Edition** (serverless), with data in **Delta tables in Unity Catalog**
- **LLM:** `databricks-qwen35-122b-a10b` via the Foundation Model API, called with a hand-written tool-calling loop
- **Tools:** `src/winback_tools.py` does the math and enforces the rules; the LLM decides
- **MLflow tracing**, deployed as a **Databricks Asset Bundle**

## Repository layout

```
databricks.yml                 bundle config and variables
resources/winback_jobs.yml     winback_data_job and winback_agent_job (serverless)
src/00_generate_data           Xorbix synthetic data generator
src/01_setup_tables.py         builds patient_summary, location_capacity, and output tables
src/02_winback_agent.py        system prompt and agent loop
src/03_simulate_outcomes.py    simulated outcomes with a holdout group
src/winback_tools.py           the agent's 9 tools
sample_data/README.md          notes on the synthetic data
```

## Data

All data is synthetic, with no names, contact details, or clinical fields. Most generated columns turned out to be random (lead conversion, payment type, `patients.status`, `churn_risk_score`), so the agent ignores them and uses only facts derived from `visits` and `appointments`.

## Measuring impact

10% of the patients the agent chooses are randomly **held out**: recorded but not contacted. Assignment is a hash of run and patient ID, so the agent can't influence it. **Lift = contacted rebook rate − holdout rebook rate.**

Outcomes are simulated from patient facts (recency, visit habit, show rate), not from the category the agent picked. The formulas are stated in `03_simulate_outcomes.py`. With real data, actual rebookings replace the simulation.

| | Rebook rate |
|---|---|
| Contacted (n = 68) | 25.0% |
| Holdout (n = 7) | 14.3% |
| **Simulated lift** | **+10.7 points** |

These are simulated outcomes with a small holdout, so treat the lift as an early signal.

## Deploy and run

Requires the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) and Git.

```bash
git clone <this-repo-url> && cd <repo-name>
databricks auth login --host https://<your-workspace-url>
databricks bundle validate
databricks bundle deploy
databricks bundle run winback_data_job    # first time only: generate data and build tables
databricks bundle run winback_agent_job   # run the agent, then simulate outcomes
```

Run `winback_agent_job` a second time to see the agent learn from the first run's outcomes.

**Another workspace:** log the CLI into it and override settings as needed. No code changes are required.

```bash
databricks bundle deploy --var="catalog=<catalog>" --var="schema=<schema>" --var="llm_endpoint=<endpoint>"
```

No secrets or workspace-specific values are stored in this repo.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `catalog` | `workspace` | Catalog for all tables |
| `schema` | `chiro_hackathon` | Schema for data and agent outputs |
| `llm_endpoint` | `databricks-qwen35-122b-a10b` | Model the agent uses |
| `max_actions` | `15` | Patients contacted per run |
| `lapsed_days` | `180` | Days without a visit to count as lapsed |
| `capacity_share` | `0.02` | Share of each clinic's weekly open slots a run may fill |

## Troubleshooting

- **Bundle not found:** the file must be named `databricks.yml`, in lowercase.
- **429 rate limit errors:** Free Edition limits. The loop retries automatically; lower `max_actions` for testing.
- **Missing `arm` column:** run `ALTER TABLE <catalog>.<schema>.winback_outcomes ADD COLUMNS (arm STRING);`
- **"No new decisions to simulate":** expected until the agent finishes a new run.

## Team

Pranjal Paudel and Bishwash Bhattarai
