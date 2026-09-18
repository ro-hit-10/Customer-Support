"""
Anomaly *candidate* generation.

Design choice: pure-statistics anomaly detection (a fixed z-score/IQR
threshold) is deterministic and can't explain itself, while a pure-LLM
"look at 500 rows and tell me what's weird" approach is unreliable and
expensive. So this module only computes rigorous statistical candidates
(IQR outliers per category, SLA breaches); it never decides what gets
reported. The *decision* to flag, the severity judgement, and the
human-readable explanation are produced by the LLM agent in
agent/tools.py, which reviews these candidates the way a senior analyst
would review a shortlist — this is the "AI checks and flags it" behaviour
the assignment asks for, without asking the LLM to do arithmetic it's bad
at.
"""
import pandas as pd

from app import config


def _iqr_outliers(series: pd.Series, multiplier: float) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + multiplier * iqr
    lower = max(q1 - multiplier * iqr, 0)
    return (series > upper) | (series < lower)


def find_candidates(df: pd.DataFrame) -> dict:
    """
    Returns raw statistical candidates in three buckets. Each candidate
    carries enough numeric context for the LLM to reason about *why* it
    is unusual, rather than just seeing "flagged: true".
    """
    df = df.copy()
    ref_time = df["created_at"].max()
    now = pd.Timestamp.utcnow().tz_localize(None)

    # --- 1. SLA breaches: high/critical tickets still open past threshold ---
    open_mask = df["status"].isin(["Open", "Escalated"])
    urgent_mask = df["priority"].isin(["High", "Critical"])
    age_hours = (ref_time - df["created_at"]).dt.total_seconds() / 3600
    sla_mask = open_mask & urgent_mask & (age_hours > config.SLA_BREACH_HOURS)

    sla_breaches = df.loc[sla_mask, [
        "ticket_id", "priority", "status", "category", "created_at", "agent_id", "issue_summary"
    ]].copy()
    sla_breaches["created_at"] = sla_breaches["created_at"].astype(str)
    sla_breaches["age_hours"] = age_hours[sla_mask].round(1)
    sla_breaches["age_days"] = (age_hours[sla_mask] / 24).round(1)
    sla_breaches = sla_breaches.sort_values("age_hours", ascending=False).head(15)

    # --- 2. Resolution-time outliers, computed per category (an 8hr
    #        resolution is normal for "General" but unusual for "Billing") ---
    resolved = df[df["resolution_time_hrs"].notna()].copy()
    outlier_frames = []
    for cat, group in resolved.groupby("category"):
        if len(group) < 5:
            continue
        mask = _iqr_outliers(group["resolution_time_hrs"], config.IQR_MULTIPLIER)
        flagged = group.loc[mask, [
            "ticket_id", "category", "priority", "agent_id",
            "resolution_time_hrs", "customer_rating", "issue_summary"
        ]].copy()
        if not flagged.empty:
            flagged["category_median_hrs"] = round(group["resolution_time_hrs"].median(), 1)
            outlier_frames.append(flagged)
    resolution_outliers = (
        pd.concat(outlier_frames).sort_values("resolution_time_hrs", ascending=False).head(15)
        if outlier_frames else pd.DataFrame()
    )

    # --- 3. Low-rating clusters per agent (possible quality issue, not a
    #        single-ticket anomaly but a pattern worth surfacing) ---
    rated = df[df["customer_rating"].notna()]
    agent_stats = rated.groupby("agent_id")["customer_rating"].agg(["mean", "count"])
    low_rating_agents = agent_stats[(agent_stats["mean"] < 3.0) & (agent_stats["count"] >= 5)]
    low_rating_agents = low_rating_agents.round(2).reset_index()

    return {
        "sla_breaches": sla_breaches.to_dict(orient="records"),
        "resolution_time_outliers": resolution_outliers.to_dict(orient="records"),
        "low_rating_agents": low_rating_agents.to_dict(orient="records"),
        "generated_at": now.isoformat(),
    }
