"""PDF report generator for Blue Lotus Labs run results (requires reportlab)."""

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

BL_NAVY = colors.HexColor("#0D1B2A")
BL_BLUE = colors.HexColor("#1B4F72")
BL_TEAL = colors.HexColor("#148F77")
BL_GOLD = colors.HexColor("#D4AC0D")
BL_ROSE = colors.HexColor("#C0392B")
BL_LIGHT = colors.HexColor("#EAF2FF")
BL_GREY = colors.HexColor("#5D6D7E")
WHITE = colors.white
BLACK = colors.black

def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold",
                                fontSize=22, textColor=WHITE, alignment=TA_CENTER,
                                spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica",
                                   fontSize=11, textColor=BL_GOLD, alignment=TA_CENTER,
                                   spaceAfter=2),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold",
                                  fontSize=13, textColor=BL_BLUE, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body", fontName="Helvetica",
                               fontSize=9, textColor=BLACK, spaceAfter=4, leading=14),
        "small": ParagraphStyle("small", fontName="Helvetica",
                                fontSize=7, textColor=BL_GREY, spaceAfter=2),
        "warning": ParagraphStyle("warning", fontName="Helvetica-Oblique",
                                  fontSize=8, textColor=BL_ROSE, spaceAfter=4),
        "metric_label": ParagraphStyle("ml", fontName="Helvetica-Bold",
                                       fontSize=9, textColor=BL_GREY),
        "metric_value": ParagraphStyle("mv", fontName="Helvetica-Bold",
                                       fontSize=16, textColor=BL_NAVY),
    }

def _pct(x, dp=2):
    """Format a fractional value as a clean percentage string.

    e.g. -0.153601 -> '-15.36%'. Returns an em dash for missing/invalid values
    so a client-facing report never shows a raw float or 'None'.
    """
    try:
        if x is None:
            return "—"
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "—"


def _metric_table(rows: list) -> Table:
    """Render a 2-column metrics table."""
    style = _styles()
    data = []
    for label, value, note in rows:
        data.append([
            Paragraph(label, style["metric_label"]),
            Paragraph(str(value), style["metric_value"]),
            Paragraph(note or "", style["small"]),
        ])
    t = Table(data, colWidths=[5.5*cm, 4*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BL_LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, BL_LIGHT]),
        ("GRID",       (0, 0), (-1, -1), 0.3, BL_GREY),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    return t

def generate_pdf(result: dict, strategy_name: str,
                 ticker: Optional[str] = None,
                 run_id: Optional[str] = None) -> bytes:
    """
    Generate a branded PDF report from a serialized engine result.

    Parameters
    ----------
    result        : dict from serialize_run_results()
    strategy_name : display name
    ticker        : optional ticker symbol
    run_id        : optional run ID for reference

    Returns
    -------
    PDF bytes — write to file or stream to client
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=1.5*cm, bottomMargin=2*cm,
                               leftMargin=2*cm, rightMargin=2*cm)
    style = _styles()
    story = []

    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    header_data = [[
        Paragraph("BLUE LOTUS LABS", style["title"]),
    ]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BL_NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Constraint-Driven Monte Carlo Stress-Testing Report", style["subtitle"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(f"Strategy: {strategy_name}" + (f"  |  Ticker: {ticker}" if ticker else ""),
                            style["subtitle"]))
    story.append(Paragraph(f"Generated: {generated}" + (f"  |  Run ID: {run_id}" if run_id else ""),
                            style["small"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BL_GOLD, spaceAfter=10))

    story.append(Paragraph("1. Data & Simulation Summary", style["section"]))
    meta = result.get("metadata", {})
    sim = result.get("simulation", {})
    sc = sim.get("scenario_counts", {})
    story.append(_metric_table([
        ("Observations",    meta.get("n_observations", "—"),    "Number of daily returns used"),
        ("Raw Mean Return", _pct(meta.get('raw_mean'), 4),      "Daily mean before processing"),
        ("Raw Std Dev",     _pct(meta.get('raw_std'), 4),       "Daily standard deviation"),
        ("Paths Simulated", f"{sim.get('n_paths', 0):,}",       "Monte Carlo paths generated"),
        ("Horizon",         f"{sim.get('horizon', 0)} days",    "Forward simulation window"),
        ("Rejection Rate",  f"{sim.get('rejection_rate', 0):.1%}", "Paths rejected by hard constraints"),
        ("Normal Paths",    f"{sc.get('normal', 0):,}",         "Max drawdown < moderate threshold"),
        ("Stress Paths",    f"{sc.get('stress', 0):,}",         "Moderate to severe drawdown"),
        ("Crisis Paths",    f"{sc.get('crisis', 0):,}",         "Severe drawdown paths"),
    ]))

    story.append(Paragraph("2. Regime Analysis", style["section"]))
    reg = result.get("regime", {})
    dist = reg.get("stationary_dist", {})
    story.append(_metric_table([
        ("Calm Regime",     f"{dist.get('calm', 0):.1%}",     "Long-run fraction in calm state"),
        ("Volatile Regime", f"{dist.get('volatile', 0):.1%}", "Long-run fraction in volatile state"),
        ("Crisis Regime",   f"{dist.get('crisis', 0):.1%}",  "Long-run fraction in crisis state"),
    ]))

    story.append(Paragraph("3. Maximum Drawdown", style["section"]))
    dd = result.get("drawdown", {})
    dd_mean_ci = (dd.get("bootstrap_ci") or {}).get("mean") or {}
    story.append(_metric_table([
        ("Mean Max Drawdown",   _pct(dd.get('mean')),   "Average worst drawdown across all paths"),
        ("Median Max Drawdown", _pct(dd.get('median')), "50th percentile drawdown"),
        ("5th Percentile",      _pct(dd.get('p5')),     "Severe tail — only 5% of paths worse"),
        ("90% CI of Mean (Lower)", _pct(dd_mean_ci.get('lo')), "Lower bound of the 90% CI for the mean"),
        ("90% CI of Mean (Upper)", _pct(dd_mean_ci.get('hi')), "Upper bound of the 90% CI for the mean"),
    ]))

    story.append(Paragraph("4. Expected Shortfall (CVaR)", style["section"]))
    es = result.get("expected_shortfall", {})
    es_agg_ci = (es.get("bootstrap_ci") or {}).get("aggregate") or {}
    story.append(_metric_table([
        ("Alpha Level",      f"{es.get('alpha', 0.05):.0%}",       "Tail probability threshold"),
        ("Aggregate ES",     _pct(es.get('aggregate')),            "Mean return in worst α% of scenarios"),
        ("Mean Per-Path ES", _pct(es.get('mean')),                 "Average per-path expected shortfall"),
        ("90% CI of Mean (Lower)", _pct(es_agg_ci.get('lo')),      "Lower bound of the 90% CI for the mean"),
        ("90% CI of Mean (Upper)", _pct(es_agg_ci.get('hi')),      "Upper bound of the 90% CI for the mean"),
    ]))

    story.append(Paragraph("5. Time-to-Recovery", style["section"]))
    rec = result.get("recovery", {})
    _horizon = sim.get('horizon', 252)
    story.append(_metric_table([
        ("Mean Recovery",   f"{round(rec['mean'])} days" if rec.get('mean') else "—",
                            "Avg days to reclaim peak — recovered paths only"),
        ("Median Recovery", f"{round(rec['median'])} days" if rec.get('median') else "—",
                            "Median days to reclaim peak — recovered paths only"),
        ("Unrecovered at Horizon", f"{rec.get('pct_never', 0):.1%}",
                            f"Still below peak when the {_horizon}-day window ends (censored, not permanent loss)"),
    ]))

    frag = result.get("fragility", {})
    if frag.get("index") is not None:
        story.append(Paragraph("6. Model Fragility Index™", style["section"]))
        story.append(Paragraph(
            "The Model Fragility Index™ measures how sensitive risk estimates are to "
            "small changes in model parameters. Lower scores indicate more robust estimates.",
            style["body"]
        ))
        fi_color = BL_TEAL if frag.get("grade") == "Robust" else (
                   BL_GOLD if frag.get("grade") == "Moderate" else BL_ROSE)
        story.append(_metric_table([
            ("Fragility Score", f"{frag.get('index', 0):.4f}", "Range 0 (robust) to 1 (fragile)"),
            ("Fragility Grade", frag.get("grade", "—"),         "Robust / Moderate / Fragile"),
        ]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BL_GREY))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠  IMPORTANT DISCLAIMER: This report estimates risk distributions only. "
        "No return predictions are made or implied. Past performance is not indicative "
        "of future results. This report is produced by Blue Lotus Labs for informational "
        "purposes only and does not constitute investment advice. Results are based on "
        "statistical modelling of historical data and are subject to model risk and "
        "data limitations.",
        style["warning"]
    ))
    story.append(Paragraph(
        f"Blue Lotus Labs  |  {generated}  |  bluelotuslabs.net",
        style["small"]
    ))

    doc.build(story)
    return buf.getvalue()
