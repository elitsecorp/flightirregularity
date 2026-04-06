from __future__ import annotations

import csv
import io
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, send_file, abort, url_for
from werkzeug.utils import secure_filename
#
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:
    REPORTLAB_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "zip"}
REQUIRED_COLUMNS = {
    "Timestamp",
    "FLIGHT_PHASE",
    "AIR_GROUND",
    "HEIGHT",
    "IAS_C",
    "GS_C",
    "IVV_C",
    "PITCH_C",
    "ROLL_C",
    "FLAPC",
    "LDG_SEL_UP",
    "AP_ENGAGED",
}


app = Flask(__name__)
app.secret_key = "flight-debrief-demo"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@dataclass
class LandingSegment:
    touchdown_idx: int
    touchdown_time: pd.Timestamp
    approach: pd.DataFrame
    flare: pd.DataFrame
    touchdown: pd.DataFrame
    landing: pd.DataFrame


@dataclass(frozen=True)
class LandingRule:
    key: str
    title: str
    statement: str
    source: str
    kind: str


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def sniff_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=";,")
        return dialect.delimiter
    except csv.Error:
        return ";"


def read_dataframe_from_path(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
            if not csv_names:
                raise ValueError("The zip file does not contain a CSV file.")
            with zf.open(csv_names[0]) as fh:
                raw = fh.read().decode("utf-8-sig", errors="replace")
        delimiter = sniff_delimiter(raw)
        df = pd.read_csv(io.StringIO(raw), sep=delimiter)
    else:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        delimiter = sniff_delimiter(raw)
        df = pd.read_csv(io.StringIO(raw), sep=delimiter)

    df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" not in df.columns:
        raise ValueError("Missing Timestamp column.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    for col in REQUIRED_COLUMNS:
        if col == "Timestamp" or col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    return df


def detect_touchdown_index(df: pd.DataFrame) -> int:
    air = pd.to_numeric(df["AIR_GROUND"], errors="coerce").ffill().fillna(0).astype(int)
    transitions = df.index[(air.shift(1) == 0) & (air == 1)].tolist()
    if transitions:
        return transitions[-1]

    ground_rows = df.index[air == 1].tolist()
    if ground_rows:
        return ground_rows[-1]

    return len(df) - 1


def segment_landing(df: pd.DataFrame) -> LandingSegment:
    touchdown_idx = detect_touchdown_index(df)
    touchdown_time = df.loc[touchdown_idx, "Timestamp"]

    pre_window = df[(df["Timestamp"] >= touchdown_time - pd.Timedelta(minutes=5)) & (df["Timestamp"] < touchdown_time)].copy()
    post_window = df[(df["Timestamp"] >= touchdown_time) & (df["Timestamp"] <= touchdown_time + pd.Timedelta(seconds=90))].copy()
    landing = pd.concat([pre_window, post_window], ignore_index=True)

    airborne = pre_window[pd.to_numeric(pre_window["AIR_GROUND"], errors="coerce").fillna(0).astype(int) == 0].copy()
    flare = airborne[airborne["HEIGHT"] <= 200].copy()
    approach = airborne[airborne["HEIGHT"] > 200].copy()

    if approach.empty and not airborne.empty:
        approach = airborne.iloc[: max(1, len(airborne) // 2)].copy()
    if flare.empty and not airborne.empty:
        flare = airborne.iloc[max(0, len(airborne) - min(60, len(airborne))):].copy()
    if post_window.empty:
        post_window = df.iloc[touchdown_idx:].copy()

    touchdown = post_window.copy()
    return LandingSegment(
        touchdown_idx=touchdown_idx,
        touchdown_time=touchdown_time,
        approach=approach,
        flare=flare,
        touchdown=touchdown,
        landing=landing,
    )


def line_trace(fig: go.Figure, frame: pd.DataFrame, y: str, row: int, name: str, color: str, dashed: bool = False) -> None:
    if frame.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=frame["Timestamp"],
            y=frame[y],
            mode="lines",
            name=name,
            line=dict(color=color, width=2, dash="dash" if dashed else "solid"),
            hovertemplate="%{x|%H:%M:%S}<br>%{y:.2f}<extra>" + name + "</extra>",
        ),
        row=row,
        col=1,
    )


def build_summary_figure(segment: LandingSegment) -> str:
    landing = segment.landing.copy()
    touchdown_time = segment.touchdown_time

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.33, 0.22, 0.22, 0.23],
        subplot_titles=("Altitude and configuration", "Airspeed and ground speed", "Vertical motion", "Attitude"),
    )

    line_trace(fig, landing, "HEIGHT", 1, "Height", "#f97316")
    line_trace(fig, landing, "FLAPC", 1, "Flaps", "#94a3b8", dashed=True)
    line_trace(fig, landing, "IAS_C", 2, "IAS", "#0f766e")
    line_trace(fig, landing, "GS_C", 2, "GS", "#2563eb", dashed=True)
    line_trace(fig, landing, "IVV_C", 3, "Vertical speed", "#b91c1c")
    line_trace(fig, landing, "VRTG_C", 3, "Load factor", "#7c3aed", dashed=True)
    line_trace(fig, landing, "PITCH_C", 4, "Pitch", "#1d4ed8")
    line_trace(fig, landing, "ROLL_C", 4, "Roll", "#c026d3", dashed=True)

    for r in range(1, 5):
        fig.add_vline(x=touchdown_time, line_width=1, line_dash="dot", line_color="#eab308", row=r, col=1)

    fig.update_layout(
        template="plotly_white",
        height=940,
        margin=dict(l=35, r=20, t=60, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#0f172a"),
    )
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Ft / units", row=1, col=1)
    fig.update_yaxes(title_text="Knots", row=2, col=1)
    fig.update_yaxes(title_text="FPM / g", row=3, col=1)
    fig.update_yaxes(title_text="Deg", row=4, col=1)

    return fig.to_html(full_html=False, include_plotlyjs=True, config={"displayModeBar": False})


def build_3d_figure(segment: LandingSegment) -> str:
    landing = segment.landing.copy()
    if landing.empty:
        return ""

    landing["phase"] = "Approach"
    landing.loc[landing["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=75), "phase"] = "Flare"
    landing.loc[landing["Timestamp"] >= segment.touchdown_time, "phase"] = "Touchdown"
    color_map = {"Approach": "#0f766e", "Flare": "#f97316", "Touchdown": "#ef4444"}

    fig = go.Figure()
    for phase, frame in landing.groupby("phase"):
        fig.add_trace(
            go.Scatter3d(
                x=(frame["Timestamp"] - landing["Timestamp"].min()).dt.total_seconds(),
                y=frame["HEIGHT"],
                z=frame["IAS_C"],
                mode="lines+markers",
                name=phase,
                marker=dict(size=3.5, color=color_map.get(phase, "#64748b")),
                line=dict(color=color_map.get(phase, "#64748b"), width=4),
                hovertemplate="t=%{x:.0f}s<br>Height=%{y:.1f}<br>IAS=%{z:.1f}<extra>" + phase + "</extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="Seconds in landing window",
            yaxis_title="Height",
            zaxis_title="IAS",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#0f172a"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


LANDING_RULES: list[LandingRule] = [
    LandingRule(
        key="stabilized_approach",
        title="Stabilized approach",
        statement="All approaches should be stabilized by 1,000 feet AFE in IMC and by 500 feet AFE in VMC. The airplane should be on the correct flight path, with only small heading and pitch changes, correct landing configuration, sink rate no greater than 1,000 fpm, thrust appropriate to configuration, and all briefings and checklists complete.",
        source="737 MAX FCTM.pdf, p. 171, 5.5 Recommended Elements of a Stabilized Approach",
        kind="stability",
    ),
    LandingRule(
        key="unstabilized_go_around",
        title="Unstabilized approach response",
        statement="If the approach becomes unstabilized below 1,000 feet AFE in IMC or below 500 feet AFE in VMC, a missed approach is recommended. If criteria cannot be established and maintained until approaching the flare, initiate a go-around.",
        source="737 MAX FCTM.pdf, p. 171-172, 5.5-5.6 Recommended Elements of a Stabilized Approach",
        kind="goaround",
    ),
    LandingRule(
        key="threshold_speed_touchdown_zone",
        title="Threshold and touchdown zone",
        statement="As the airplane crosses the runway threshold it should be stabilized on approach airspeed to within +10 knots until arresting descent at flare and positioned to make a normal landing in the touchdown zone, defined as the first 3,000 feet or first third of the runway, whichever is less.",
        source="737 MAX FCTM.pdf, p. 172, 5.6 Recommended Elements of a Stabilized Approach",
        kind="threshold",
    ),
    LandingRule(
        key="ils_landing_config",
        title="ILS landing configuration",
        statement="At glideslope alive, the SOP calls for gear down, flaps 15, speedbrake arm, engine start switches CONT, landing checklist, and the missed-approach altitude set on the MCP later in the approach. For a single-channel approach, the autopilot and autothrottle are to be disconnected no later than the minimum use height.",
        source="B737 SOP 2024 Rev8.pdf, p. 65-66, 2.13 Landing Procedure ILS",
        kind="config",
    ),
    LandingRule(
        key="flare_timing",
        title="Flare initiation",
        statement="Initiate the flare when the main gear is approximately 20 feet above the runway by increasing pitch attitude approximately 2 to 3 degrees. Use smooth, small pitch adjustments and do not trim during the flare.",
        source="737 MAX FCTM.pdf, p. 259-260, 6.7-6.8 Flare and Touchdown",
        kind="flare",
    ),
    LandingRule(
        key="landing_roll",
        title="Landing roll",
        statement="After touchdown, fly the nose wheels smoothly onto the runway without delay, avoid holding the nose wheel off, and maintain runway alignment. Speedbrakes should extend automatically and, if not, be extended manually without delay.",
        source="737 MAX FCTM.pdf, p. 277-278, 6.25-6.26 Landing Roll",
        kind="roll",
    ),
]


def derive_landing_metrics(segment: LandingSegment) -> dict[str, float | int | pd.Timestamp | None]:
    df = segment.landing.copy()
    touchdown = segment.touchdown.copy()
    last_airborne = df[df["Timestamp"] < segment.touchdown_time]
    airborne_30 = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=30)]
    airborne_20 = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=20)]
    airborne_10 = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=10)]

    metrics = {
        "approach_speed_mean": safe_mean(airborne_30["IAS_C"]),
        "approach_speed_std": safe_std(airborne_30["IAS_C"]),
        "approach_sink_max": float((-pd.to_numeric(airborne_30["IVV_C"], errors="coerce")).dropna().max()) if not airborne_30.empty else float("nan"),
        "approach_roll_max": float(pd.to_numeric(airborne_30["ROLL_C"], errors="coerce").dropna().abs().max()) if not airborne_30.empty else float("nan"),
        "touchdown_pitch": float(pd.to_numeric(touchdown["PITCH_C"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else float("nan"),
        "touchdown_roll": float(pd.to_numeric(touchdown["ROLL_C"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else float("nan"),
        "touchdown_speed": float(pd.to_numeric(touchdown["IAS_C"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else float("nan"),
        "touchdown_flap": int(pd.to_numeric(touchdown["FLAPC"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else None,
        "touchdown_gear": int(pd.to_numeric(touchdown["LDG_SEL_UP"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else None,
    }

    ap_slice = df[df["Timestamp"] >= segment.touchdown_time - pd.Timedelta(minutes=5)]
    ap_last_on = ap_slice[ap_slice["AP_ENGAGED"] == 1]["Timestamp"].max() if not ap_slice.empty else pd.NaT
    metrics["ap_disconnect_seconds"] = (
        (segment.touchdown_time - ap_last_on).total_seconds() if pd.notna(ap_last_on) else None
    )
    metrics["ap_last_on"] = ap_last_on if pd.notna(ap_last_on) else None

    flare = None
    pitch = pd.to_numeric(airborne_20["PITCH_C"], errors="coerce")
    height = pd.to_numeric(airborne_20["HEIGHT"], errors="coerce")
    if not airborne_20.empty:
        baseline = pitch.rolling(5, min_periods=1).median().shift(1)
        flare_candidates = airborne_20[(height <= 60) & ((pitch - baseline) >= 1.5)]
        if not flare_candidates.empty:
            flare = flare_candidates.iloc[0]
        else:
            low_height = airborne_20[height <= 30]
            if not low_height.empty:
                flare = low_height.iloc[0]
    metrics["flare_start_time"] = flare["Timestamp"] if flare is not None else None
    metrics["flare_start_height"] = float(flare["HEIGHT"]) if flare is not None else float("nan")
    metrics["flare_start_pitch"] = float(flare["PITCH_C"]) if flare is not None else float("nan")
    metrics["flare_sink_rate"] = float(-flare["IVV_C"]) if flare is not None else float("nan")
    return metrics


def evaluate_rule(rule: LandingRule, metrics: dict[str, float | int | pd.Timestamp | None]) -> dict[str, str]:
    status = "na"
    rationale = "Not enough data to assess."

    if rule.kind == "stability":
        sink = metrics["approach_sink_max"]
        speed_std = metrics["approach_speed_std"]
        roll = metrics["approach_roll_max"]
        touch_flap = metrics["touchdown_flap"]
        touch_gear = metrics["touchdown_gear"]
        checks = []
        if not pd.isna(sink):
            checks.append(f"sink {sink:.0f} fpm")
        if not pd.isna(speed_std):
            checks.append(f"speed std {speed_std:.1f} kt")
        if not pd.isna(roll):
            checks.append(f"bank {roll:.1f} deg")
        if touch_flap is not None:
            checks.append(f"flap {touch_flap}")
        if touch_gear is not None:
            checks.append("gear down" if touch_gear == 0 else "gear up")
        if not pd.isna(sink) and sink <= 1000 and not pd.isna(roll) and roll <= 5 and touch_gear == 0:
            status = "pass"
            rationale = "Final-phase sink rate and bank stayed within the stabilized-approach guardrails, and the landing gear was down."
            if not pd.isna(speed_std) and speed_std > 2:
                status = "partial"
                rationale = "Sink rate and bank were acceptable, but speed control was a little variable in the final 30 seconds."
        else:
            status = "partial"
            rationale = "The last airborne segment stayed near the stabilized-approach limits, but speed control or bank/sink margins were not fully ideal."
        return {"status": status, "rationale": rationale, "details": ", ".join(checks)}

    if rule.kind == "goaround":
        sink = metrics["approach_sink_max"]
        flare_height = metrics["flare_start_height"]
        if not pd.isna(sink) and sink > 1000:
            status = "fail"
            rationale = "Sink rate exceeded the stabilized-approach limit and would justify a go-around under the cited guidance."
        elif not pd.isna(flare_height) and flare_height < 10:
            status = "partial"
            rationale = "The flare appears late, so the approach was close to the point where the guidance recommends abandoning the landing if stability is not maintained."
        else:
            status = "pass"
            rationale = "The sampled landing stayed inside the basic stabilization envelope enough to continue."
        return {"status": status, "rationale": rationale, "details": ""}

    if rule.kind == "threshold":
        speed_std = metrics["approach_speed_std"]
        touchdown_speed = metrics["touchdown_speed"]
        if not pd.isna(speed_std) and speed_std <= 2 and not pd.isna(touchdown_speed):
            status = "pass"
            rationale = "Speed stayed fairly steady right before touchdown, but runway position cannot be measured without lateral/threshold data."
        else:
            status = "partial"
            rationale = "The landing speed profile is acceptable but the exact touchdown-zone requirement cannot be verified from this file."
        return {"status": status, "rationale": rationale, "details": ""}

    if rule.kind == "config":
        touch_flap = metrics["touchdown_flap"]
        touch_gear = metrics["touchdown_gear"]
        ap_disconnect_seconds = metrics["ap_disconnect_seconds"]
        parts = []
        if touch_gear == 0:
            parts.append("gear down at touchdown")
        else:
            parts.append("gear status unclear")
        if touch_flap is not None:
            parts.append(f"flaps {touch_flap} at touchdown")
        if ap_disconnect_seconds is not None:
            parts.append(f"AP off {ap_disconnect_seconds:.0f} s before touchdown")
        if touch_gear == 0 and touch_flap in {15, 30, 40}:
            status = "pass"
            rationale = "The landing configuration was complete by touchdown, and the autopilot was already off well before flare."
        else:
            status = "partial"
            rationale = "The file shows landing configuration at touchdown, but not every procedural timing point can be verified from the telemetry."
        return {"status": status, "rationale": rationale, "details": ", ".join(parts)}

    if rule.kind == "flare":
        flare_height = metrics["flare_start_height"]
        flare_pitch = metrics["flare_start_pitch"]
        touchdown_pitch = metrics["touchdown_pitch"]
        touchdown_roll = metrics["touchdown_roll"]
        if not pd.isna(flare_height) and 10 <= flare_height <= 30:
            status = "pass"
            rationale = f"The flare starts around {flare_height:.1f} ft, which is broadly in line with the cited flare initiation range."
        elif not pd.isna(flare_height) and flare_height < 10:
            status = "fail"
            rationale = f"The flare appears to start around {flare_height:.1f} ft, much later than the approximate 20 ft guidance."
        elif not pd.isna(flare_height):
            status = "partial"
            rationale = f"The flare appears to start around {flare_height:.1f} ft, which is offset from the approximate 20 ft guidance."
        else:
            status = "partial"
            rationale = "Flare timing could not be isolated cleanly from the telemetry."
        details = []
        if not pd.isna(flare_pitch):
            details.append(f"flare pitch {flare_pitch:.1f} deg")
        if not pd.isna(touchdown_pitch):
            details.append(f"touchdown pitch {touchdown_pitch:.1f} deg")
        if not pd.isna(touchdown_roll):
            details.append(f"touchdown roll {touchdown_roll:.1f} deg")
        return {"status": status, "rationale": rationale, "details": ", ".join(details)}

    if rule.kind == "roll":
        touchdown_pitch = metrics["touchdown_pitch"]
        touchdown_roll = metrics["touchdown_roll"]
        if not pd.isna(touchdown_pitch) and abs(touchdown_pitch) <= 3 and not pd.isna(touchdown_roll) and abs(touchdown_roll) <= 5:
            status = "pass"
            rationale = "Touchdown attitude was modest and the airplane was not banked significantly at the runway contact point."
        else:
            status = "partial"
            rationale = "The landing roll cannot be fully checked without speedbrake and nosewheel data, but the touchdown attitude looks reasonable."
        return {"status": status, "rationale": rationale, "details": ""}

    return {"status": status, "rationale": rationale, "details": ""}


def build_rule_assessment(segment: LandingSegment) -> dict:
    metrics = derive_landing_metrics(segment)
    rule_checks = []
    findings = []
    improvements = []
    scores = {"pass": 18, "partial": 8, "fail": 0, "na": 6}
    total = 0
    max_total = 0

    for rule in LANDING_RULES:
        result = evaluate_rule(rule, metrics)
        rule_checks.append(
            {
                "title": rule.title,
                "statement": rule.statement,
                "source": rule.source,
                "status": result["status"],
                "rationale": result["rationale"],
                "details": result.get("details", ""),
            }
        )
        weight = 18 if rule.kind in {"stability", "goaround", "flare", "config"} else 12
        max_total += weight
        total += scores.get(result["status"], 0)

        if result["status"] in {"fail", "partial"}:
            findings.append(f"{rule.title}: {result['rationale']} ({rule.source})")
        if rule.kind == "flare" and result["status"] != "pass":
            improvements.append("Initiate the flare earlier, closer to the 20 ft main-gear cue described in the FCTM.")
        elif rule.kind == "stability" and result["status"] != "pass":
            improvements.append("Tighten energy management in the last 30 seconds so the approach is fully stabilized before the flare.")
        elif rule.kind == "roll" and result["status"] != "pass":
            improvements.append("After touchdown, keep the pitch from increasing and deploy speedbrakes promptly.")
        elif rule.kind == "config" and result["status"] != "pass":
            improvements.append("Verify the landing configuration earlier so gear, flaps, and AP state are all correct before the flare.")

    score = round((total / max_total) * 100) if max_total else 0
    if score >= 85:
        overall = "Stable"
    elif score >= 70:
        overall = "Moderate"
    else:
        overall = "Needs review"

    if not findings:
        findings = ["The landing satisfies the extracted rule set in the measurable data."]
    if not improvements:
        improvements = ["Continue the current technique and keep using the same rule set to crosscheck future landings."]

    return {
        "overall": overall,
        "score": score,
        "metrics": [
            {"label": "Approach speed", "value": f"{metrics['approach_speed_mean']:.1f} kt" if not pd.isna(metrics["approach_speed_mean"]) else "n/a"},
            {"label": "Speed variation", "value": f"{metrics['approach_speed_std']:.1f} kt" if not pd.isna(metrics["approach_speed_std"]) else "n/a"},
            {"label": "Touchdown speed", "value": f"{metrics['touchdown_speed']:.1f} kt" if not pd.isna(metrics["touchdown_speed"]) else "n/a"},
            {"label": "Touchdown flap", "value": str(metrics["touchdown_flap"]) if metrics["touchdown_flap"] is not None else "n/a"},
            {"label": "Touchdown gear", "value": "Down" if metrics["touchdown_gear"] == 0 else "Check" if metrics["touchdown_gear"] is not None else "n/a"},
            {"label": "Flare start", "value": f"{metrics['flare_start_height']:.1f} ft" if not pd.isna(metrics["flare_start_height"]) else "n/a"},
            {"label": "AP off before touchdown", "value": f"{metrics['ap_disconnect_seconds']:.0f} s" if metrics["ap_disconnect_seconds"] is not None else "n/a"},
        ],
        "findings": findings,
        "improvements": improvements,
        "rule_checks": rule_checks,
        "metrics_raw": metrics,
    }


def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0) -> list[list[float]]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def transform_point(point: tuple[float, float, float], matrix: list[list[float]], origin: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    ox, oy, oz = origin
    return (
        ox + matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        oy + matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        oz + matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def cuboid_mesh(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> tuple[list[tuple[float, float, float]], list[int], list[int], list[int]]:
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    i = [0, 0, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3]
    j = [1, 2, 5, 6, 4, 5, 6, 7, 3, 2, 6, 7]
    k = [2, 3, 6, 7, 5, 6, 7, 4, 1, 5, 7, 4]
    return vertices, i, j, k


def make_mesh_trace(
    vertices: list[tuple[float, float, float]],
    i: list[int],
    j: list[int],
    k: list[int],
    origin: tuple[float, float, float],
    matrix: list[list[float]],
    color: str,
    opacity: float,
    name: str | None = None,
    showscale: bool = False,
) -> go.Mesh3d:
    pts = [transform_point(pt, matrix, origin) for pt in vertices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showscale=showscale,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.12, roughness=0.8),
    )


def build_aircraft_model(origin: tuple[float, float, float], pitch_deg: float, roll_deg: float, yaw_deg: float, opacity: float, name_suffix: str = "") -> list[go.Mesh3d]:
    matrix = rotation_matrix(roll_deg, pitch_deg, yaw_deg)

    fuselage_v, fuselage_i, fuselage_j, fuselage_k = cuboid_mesh(-58, 58, -4.5, 4.5, -4.3, 4.3)
    wing_v, wing_i, wing_j, wing_k = cuboid_mesh(-8, 10, -52, 52, -1.1, 1.1)
    tailplane_v, tailplane_i, tailplane_j, tailplane_k = cuboid_mesh(40, 56, -18, 18, -0.8, 0.8)
    fin_v, fin_i, fin_j, fin_k = cuboid_mesh(44, 52, -1.6, 1.6, -1.0, 15.0)
    engine_v, engine_i, engine_j, engine_k = cuboid_mesh(-18, -6, -31, -21, -3.3, -1.0)
    engine2_v, engine2_i, engine2_j, engine2_k = cuboid_mesh(-18, -6, 21, 31, -3.3, -1.0)

    return [
        make_mesh_trace(fuselage_v, fuselage_i, fuselage_j, fuselage_k, origin, matrix, "#cbd5e1", opacity, name=f"Fuselage{name_suffix}"),
        make_mesh_trace(wing_v, wing_i, wing_j, wing_k, origin, matrix, "#94a3b8", opacity * 0.95, name=f"Wing{name_suffix}"),
        make_mesh_trace(tailplane_v, tailplane_i, tailplane_j, tailplane_k, origin, matrix, "#94a3b8", opacity * 0.92, name=f"Tailplane{name_suffix}"),
        make_mesh_trace(fin_v, fin_i, fin_j, fin_k, origin, matrix, "#cbd5e1", opacity * 0.9, name=f"Fin{name_suffix}"),
        make_mesh_trace(engine_v, engine_i, engine_j, engine_k, origin, matrix, "#64748b", opacity * 0.95, name=f"Engine L{name_suffix}"),
        make_mesh_trace(engine2_v, engine2_i, engine2_j, engine2_k, origin, matrix, "#64748b", opacity * 0.95, name=f"Engine R{name_suffix}"),
    ]


def build_replay_assessment(row: pd.Series, touchdown_time: pd.Timestamp) -> dict[str, str]:
    ts = pd.to_datetime(row.get("Timestamp"), errors="coerce")
    height = pd.to_numeric(pd.Series([row.get("HEIGHT")]), errors="coerce").iloc[0]
    bank = pd.to_numeric(pd.Series([row.get("ROLL_C")]), errors="coerce").iloc[0]
    pitch = pd.to_numeric(pd.Series([row.get("PITCH_C")]), errors="coerce").iloc[0]
    sink = max(0.0, -pd.to_numeric(pd.Series([row.get("IVV_C")]), errors="coerce").iloc[0])
    g = pd.to_numeric(pd.Series([row.get("VRTG_C")]), errors="coerce").iloc[0]
    engine_vals = []
    for candidate in ("N11_C", "N12_C", "TORQ1", "TORQ2", "NP1", "NP2"):
        if candidate in row.index:
            value = pd.to_numeric(pd.Series([row.get(candidate)]), errors="coerce").iloc[0]
            if pd.notna(value):
                engine_vals.append((candidate, float(value)))

    reasons: list[str] = []
    severity = "green"

    if pd.notna(bank):
        if abs(bank) > 6:
            severity = "red"
            reasons.append(f"bank {abs(bank):.1f} deg > 6 deg red")
        elif abs(bank) > 4 and severity != "red":
            severity = "amber"
            reasons.append(f"bank {abs(bank):.1f} deg > 4 deg amber")

    if pd.notna(sink):
        if sink > 1200:
            severity = "red"
            reasons.append(f"descent rate {sink:.0f} fpm > 1200 fpm red")
        elif sink > 1000 and severity != "red":
            severity = "amber"
            reasons.append(f"descent rate {sink:.0f} fpm > 1000 fpm amber")
        if pd.notna(height) and height > 50 and sink < 200 and severity != "red":
            severity = "amber"
            reasons.append(f"descent rate {sink:.0f} fpm < 200 fpm above 50 ft amber")

    if pd.notna(pitch) and pd.notna(ts) and ts >= touchdown_time and pitch > 6:
        severity = "red"
        reasons.append(f"pitch {pitch:.1f} deg > 6 deg after touchdown red")

    if pd.notna(g) and pd.notna(height) and height <= 1000:
        if g > 2:
            severity = "red"
            reasons.append(f"vertical g {g:.2f} > 2.00 red below 1000 ft")
        elif g > 1.8 and severity != "red":
            severity = "amber"
            reasons.append(f"vertical g {g:.2f} > 1.80 amber below 1000 ft")

    if pd.notna(ts) and ts >= touchdown_time and engine_vals:
        # Use the strongest available engine column as the thrust proxy.
        max_engine = max(value for _, value in engine_vals)
        max_engine_name = max(engine_vals, key=lambda item: item[1])[0]
        if max_engine > 50 and severity != "red":
            severity = "amber"
            reasons.append(f"{max_engine_name} {max_engine:.1f} > 50 at or after touchdown amber")

    return {
        "severity": severity,
        "reason": "; ".join(reasons) if reasons else "within envelope",
    }


def build_runway_replay_figure(segment: LandingSegment) -> str:
    landing = segment.landing.copy()
    replay = landing[(landing["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=10)) & (landing["Timestamp"] <= segment.touchdown_time + pd.Timedelta(seconds=2))].copy()
    if replay.empty:
        return ""

    replay = replay.sort_values("Timestamp").reset_index(drop=True)
    timestamps = replay["Timestamp"].tolist()
    gspeed = pd.to_numeric(replay["GS_C"], errors="coerce").ffill().bfill().fillna(0.0)
    height = pd.to_numeric(replay["HEIGHT"], errors="coerce").ffill().bfill().fillna(0.0)
    pitch = pd.to_numeric(replay["PITCH_C"], errors="coerce").fillna(0.0)
    roll = pd.to_numeric(replay["ROLL_C"], errors="coerce").fillna(0.0)
    heading = pd.to_numeric(replay["HEAD_MAG"], errors="coerce").ffill().bfill()

    positions = [0.0] * len(replay)
    for idx in range(len(replay) - 2, -1, -1):
        dt = max(0.0, (timestamps[idx + 1] - timestamps[idx]).total_seconds())
        avg_speed = ((gspeed.iloc[idx] + gspeed.iloc[idx + 1]) / 2.0) * 1.68781
        positions[idx] = positions[idx + 1] - avg_speed * dt

    fig = go.Figure()

    runway_half_length = 4500
    runway_half_width = 140
    runway_x = [-runway_half_length, runway_half_length]
    runway_y = [-runway_half_width, runway_half_width]
    runway_z = [[0, 0], [0, 0]]

    fig.add_trace(
        go.Surface(
            x=[runway_x, runway_x],
            y=[[runway_y[0], runway_y[1]], [runway_y[0], runway_y[1]]],
            z=runway_z,
            showscale=False,
            opacity=0.88,
            colorscale=[[0, "#0f172a"], [1, "#334155"]],
            name="Runway",
            hoverinfo="skip",
        )
    )

    center_x = [-runway_half_length, runway_half_length]
    center_y = [0, 0]
    center_z = [0.12, 0.12]
    fig.add_trace(
        go.Scatter3d(
            x=center_x,
            y=center_y,
            z=center_z,
            mode="lines",
            line=dict(color="#e2e8f0", width=6, dash="dash"),
            name="Centerline",
            hoverinfo="skip",
        )
    )

    assessments = [build_replay_assessment(row, segment.touchdown_time) for _, row in replay.iterrows()]
    path_colors = []
    hover_text = []
    for idx, assessment in enumerate(assessments):
        severity = assessment["severity"]
        reason = assessment["reason"]
        if severity == "red":
            path_colors.append("#ef4444")
        elif severity == "amber":
            path_colors.append("#f59e0b")
        else:
            path_colors.append("#22c55e")
        hover_text.append(
            f"{timestamps[idx].strftime('%H:%M:%S')}<br>"
            f"Height: {float(height.iloc[idx]):.1f} ft<br>"
            f"Pitch: {float(pitch.iloc[idx]):.1f} deg<br>"
            f"Bank: {float(roll.iloc[idx]):.1f} deg<br>"
            f"Sink: {max(0.0, -float(pd.to_numeric(pd.Series([replay.loc[idx, 'IVV_C']]), errors='coerce').iloc[0])):.0f} fpm<br>"
            f"G: {float(pd.to_numeric(pd.Series([replay.loc[idx, 'VRTG_C']]), errors='coerce').iloc[0]):.2f}<br>"
            f"<b>{reason}</b>"
        )

    fig.add_trace(
        go.Scatter3d(
            x=positions,
            y=[0.0] * len(replay),
            z=height,
            mode="lines",
            line=dict(color="#38bdf8", width=6),
            name="CG path",
            hovertemplate="t=%{x:.0f} ft<br>Height=%{z:.1f} ft<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=positions,
            y=[0.0] * len(replay),
            z=height,
            mode="markers",
            marker=dict(size=6, color=path_colors, symbol="circle"),
            name="Deviation markers",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    amber_x, amber_y, amber_z, amber_text = [], [], [], []
    red_x, red_y, red_z, red_text = [], [], [], []
    for idx, (pos, h, assessment) in enumerate(zip(positions, height, assessments)):
        if assessment["severity"] == "amber":
            amber_x.append(pos)
            amber_y.append(0.0)
            amber_z.append(float(h))
            amber_text.append(
                f"{timestamps[idx].strftime('%H:%M:%S')}<br>"
                f"Height: {float(height.iloc[idx]):.1f} ft<br>"
                f"Pitch: {float(pitch.iloc[idx]):.1f} deg<br>"
                f"Bank: {float(roll.iloc[idx]):.1f} deg<br>"
                f"Sink: {max(0.0, -float(pd.to_numeric(pd.Series([replay.loc[idx, 'IVV_C']]), errors='coerce').iloc[0])):.0f} fpm<br>"
                f"G: {float(pd.to_numeric(pd.Series([replay.loc[idx, 'VRTG_C']]), errors='coerce').iloc[0]):.2f}<br>"
                f"<b>{assessment['reason']}</b>"
            )
        elif assessment["severity"] == "red":
            red_x.append(pos)
            red_y.append(0.0)
            red_z.append(float(h))
            red_text.append(
                f"{timestamps[idx].strftime('%H:%M:%S')}<br>"
                f"Height: {float(height.iloc[idx]):.1f} ft<br>"
                f"Pitch: {float(pitch.iloc[idx]):.1f} deg<br>"
                f"Bank: {float(roll.iloc[idx]):.1f} deg<br>"
                f"Sink: {max(0.0, -float(pd.to_numeric(pd.Series([replay.loc[idx, 'IVV_C']]), errors='coerce').iloc[0])):.0f} fpm<br>"
                f"G: {float(pd.to_numeric(pd.Series([replay.loc[idx, 'VRTG_C']]), errors='coerce').iloc[0]):.2f}<br>"
                f"<b>{assessment['reason']}</b>"
            )

    if amber_x:
        fig.add_trace(
            go.Scatter3d(
                x=amber_x,
                y=amber_y,
                z=amber_z,
                mode="markers",
                marker=dict(size=7, color="#f59e0b", symbol="diamond"),
                name="Amber deviation",
                text=amber_text,
                hovertemplate="%{text}<extra>Amber deviation</extra>",
            )
        )
    if red_x:
        fig.add_trace(
            go.Scatter3d(
                x=red_x,
                y=red_y,
                z=red_z,
                mode="markers",
                marker=dict(size=8, color="#ef4444", symbol="x"),
                name="Red deviation",
                text=red_text,
                hovertemplate="%{text}<extra>Red deviation</extra>",
            )
        )

    static_trace_count = len(fig.data)

    current_point = go.Scatter3d(
        x=[positions[0]],
        y=[0.0],
        z=[float(height.iloc[0])],
        mode="markers",
        marker=dict(size=10, color="#22c55e"),
        name="Replay cursor",
        hoverinfo="skip",
    )
    fig.add_trace(current_point)
    plane_start_index = len(fig.data)
    plane_traces = aircraft_trace_list(
        origin=(positions[0], 0.0, float(height.iloc[0])),
        pitch_deg=float(pitch.iloc[0]),
        roll_deg=float(roll.iloc[0]),
        yaw_deg=0.0,
        opacity=1.0,
        suffix=" live",
    )
    for trace in plane_traces:
        fig.add_trace(trace)

    frame_indices = list(range(len(replay)))
    frames = []
    animated_traces = [static_trace_count] + list(range(plane_start_index, plane_start_index + len(plane_traces)))
    for idx in frame_indices:
        yaw = float(heading.iloc[idx] - heading.iloc[-1]) if pd.notna(heading.iloc[idx]) and pd.notna(heading.iloc[-1]) else 0.0
        origin = (positions[idx], 0.0, float(height.iloc[idx]))
        plane = aircraft_trace_list(
            origin=origin,
            pitch_deg=float(pitch.iloc[idx]),
            roll_deg=float(roll.iloc[idx]),
            yaw_deg=yaw,
            opacity=1.0,
            suffix=f" frame {idx}",
        )
        current_color = "#ef4444" if assessments[idx]["severity"] == "red" else "#f59e0b" if assessments[idx]["severity"] == "amber" else "#22c55e"
        current_point_frame = go.Scatter3d(
            x=[positions[idx]],
            y=[0.0],
            z=[float(height.iloc[idx])],
            mode="markers",
            marker=dict(size=10, color=current_color),
            showlegend=False,
            hoverinfo="skip",
        )
        frames.append(go.Frame(data=[current_point_frame] + plane, name=str(idx), traces=animated_traces))

    fig.frames = frames

    steps = [
        dict(
            method="animate",
            label=str(idx + 1),
            args=[
                [str(idx)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 250, "redraw": True},
                    "transition": {"duration": 150},
                },
            ],
        )
        for idx in frame_indices
    ]

    fig.update_layout(
        template="plotly_dark",
        height=760,
        margin=dict(l=0, r=0, t=25, b=0),
        scene=dict(
            xaxis=dict(title="Distance to touchdown (ft)", backgroundcolor="#0b1220", gridcolor="#334155", zerolinecolor="#475569"),
            yaxis=dict(title="Lateral offset (ft)", backgroundcolor="#0b1220", gridcolor="#334155", zerolinecolor="#475569"),
            zaxis=dict(title="Height (ft)", backgroundcolor="#0b1220", gridcolor="#334155", zerolinecolor="#475569"),
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=0.75, z=0.5),
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif"),
        paper_bgcolor="#08111f",
        plot_bgcolor="#08111f",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "direction": "left",
                "x": 0.0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 250, "redraw": True}, "transition": {"duration": 150}, "fromcurrent": True, "mode": "immediate"},
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 30},
                "x": 0.05,
                "len": 0.9,
                "currentvalue": {"prefix": "Replay second: ", "font": {"size": 12}},
                "steps": steps,
            }
        ],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").dropna().mean())


def safe_std(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").dropna().std())


def build_briefing(segment: LandingSegment) -> dict:
    df = segment.landing.copy()
    touchdown = segment.touchdown.copy()
    approach = segment.approach.copy()
    flare = segment.flare.copy()

    last_airborne = df[df["Timestamp"] < segment.touchdown_time]
    touchdown_10s = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=10)]
    touchdown_30s = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=30)]

    approach_speed_mean = safe_mean(approach["IAS_C"])
    approach_speed_std = safe_std(approach["IAS_C"])
    flare_height_max = float(pd.to_numeric(flare["HEIGHT"], errors="coerce").dropna().max()) if not flare.empty else float("nan")
    sink_rate = float(-pd.to_numeric(touchdown_10s["IVV_C"], errors="coerce").dropna().min()) if not touchdown_10s.empty else float("nan")
    roll_max = float(pd.to_numeric(touchdown_30s["ROLL_C"], errors="coerce").dropna().abs().max()) if not touchdown_30s.empty else float("nan")
    pitch_max = float(pd.to_numeric(touchdown_30s["PITCH_C"], errors="coerce").dropna().max()) if not touchdown_30s.empty else float("nan")

    config = df[df["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=120)]
    ap_last_on = config[config["AP_ENGAGED"] == 1]["Timestamp"].max() if not config.empty else pd.NaT
    ap_minutes_before = ((segment.touchdown_time - ap_last_on).total_seconds() / 60) if pd.notna(ap_last_on) else None
    flap_touchdown = int(pd.to_numeric(touchdown["FLAPC"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else None
    gear_touchdown = int(pd.to_numeric(touchdown["LDG_SEL_UP"], errors="coerce").dropna().iloc[0]) if not touchdown.empty else None

    findings = []
    improvements = []
    score = 100

    if not pd.isna(sink_rate) and sink_rate > 700:
        findings.append(f"Peak sink rate near touchdown was about {sink_rate:.0f} fpm.")
        improvements.append("Reduce sink rate in the last seconds before touchdown.")
        score -= 18

    if not pd.isna(flare_height_max) and flare_height_max > 200:
        findings.append(f"The flare began around {flare_height_max:.0f} ft AGL, which is on the high side.")
        improvements.append("Start the flare lower and let the aircraft settle more progressively.")
        score -= 14

    if not pd.isna(approach_speed_std) and approach_speed_std > 3:
        findings.append(f"Approach speed varied by about {approach_speed_std:.1f} kt, which suggests some instability.")
        improvements.append("Tighten speed control on final approach.")
        score -= 12

    if not pd.isna(roll_max) and roll_max > 5:
        findings.append(f"Bank angle reached roughly {roll_max:.1f} degrees in the final 30 seconds.")
        improvements.append("Keep wings level more consistently through the flare and touchdown.")
        score -= 10

    if ap_minutes_before is not None and ap_minutes_before < 2:
        findings.append(f"Autopilot remained engaged until about {ap_minutes_before * 60:.0f} seconds before touchdown.")
        improvements.append("If SOP requires manual handling, disconnect the autopilot earlier on final.")
        score -= 10

    if gear_touchdown != 0:
        findings.append("Landing gear was not clearly down at touchdown in the data.")
        improvements.append("Verify gear-down configuration before landing.")
        score -= 12

    if flap_touchdown is not None and flap_touchdown < 20:
        findings.append(f"Flap setting at touchdown was only {flap_touchdown}, which is low for a normal landing profile.")
        improvements.append("Check landing flap selection earlier in the approach.")
        score -= 10

    if not findings:
        findings.append("The landing window looks stable in the sampled data.")
        improvements.append("Keep the current technique and confirm with cross-check metrics from additional flights.")

    score = max(0, min(100, score))
    overall = "Stable" if score >= 85 else "Moderate" if score >= 70 else "Needs review"

    if not pd.isna(approach_speed_mean):
        findings.insert(0, f"Average approach speed in the analyzed window was {approach_speed_mean:.1f} kt.")

    return {
        "overall": overall,
        "score": score,
        "metrics": [
            {"label": "Approach speed", "value": f"{approach_speed_mean:.1f} kt" if not pd.isna(approach_speed_mean) else "n/a"},
            {"label": "Speed variation", "value": f"{approach_speed_std:.1f} kt" if not pd.isna(approach_speed_std) else "n/a"},
            {"label": "Flare height", "value": f"{flare_height_max:.0f} ft" if not pd.isna(flare_height_max) else "n/a"},
            {"label": "Peak sink", "value": f"{sink_rate:.0f} fpm" if not pd.isna(sink_rate) else "n/a"},
            {"label": "Max bank", "value": f"{roll_max:.1f} deg" if not pd.isna(roll_max) else "n/a"},
            {"label": "Touchdown flap", "value": str(flap_touchdown) if flap_touchdown is not None else "n/a"},
            {"label": "Touchdown gear", "value": "Down" if gear_touchdown == 0 else "Check" if gear_touchdown is not None else "n/a"},
        ],
        "findings": findings,
        "improvements": improvements,
    }


def build_replay_metrics(segment: LandingSegment) -> list[dict[str, str]]:
    replay = segment.landing[(segment.landing["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=10)) & (segment.landing["Timestamp"] <= segment.touchdown_time)].copy()
    if replay.empty:
        return []

    replay["PITCH_C"] = pd.to_numeric(replay["PITCH_C"], errors="coerce")
    replay["ROLL_C"] = pd.to_numeric(replay["ROLL_C"], errors="coerce")
    replay["VRTG_C"] = pd.to_numeric(replay["VRTG_C"], errors="coerce")
    replay["IVV_C"] = pd.to_numeric(replay["IVV_C"], errors="coerce")
    replay["IAS_C"] = pd.to_numeric(replay["IAS_C"], errors="coerce")

    return [
        {"label": "Max bank", "value": f"{replay['ROLL_C'].abs().max():.1f} deg"},
        {"label": "Max pitch", "value": f"{replay['PITCH_C'].max():.1f} deg"},
        {"label": "Peak load factor", "value": f"{replay['VRTG_C'].max():.2f} g"},
        {"label": "Peak sink", "value": f"{(-replay['IVV_C'].min()):.0f} fpm"},
        {"label": "Touchdown speed", "value": f"{replay['IAS_C'].iloc[-1]:.1f} kt"},
    ]


def get_numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def classify_replay_point(row: pd.Series) -> tuple[str, str]:
    sink = float(max(0.0, -pd.to_numeric(pd.Series([row.get("IVV_C")]), errors="coerce").iloc[0]))
    roll = abs(float(pd.to_numeric(pd.Series([row.get("ROLL_C")]), errors="coerce").iloc[0]))
    pitch = abs(float(pd.to_numeric(pd.Series([row.get("PITCH_C")]), errors="coerce").iloc[0]))
    g = float(pd.to_numeric(pd.Series([row.get("VRTG_C")]), errors="coerce").iloc[0])

    if sink >= 1000 or roll >= 7 or g >= 1.35 or pitch >= 6:
        return "red", "red deviation"
    if sink >= 700 or roll >= 3.5 or g >= 1.2 or pitch >= 4:
        return "amber", "amber deviation"
    return "green", "within envelope"


def build_control_technique(segment: LandingSegment) -> dict:
    df = segment.landing.copy()
    touchdown = segment.touchdown.copy()
    last_airborne = df[df["Timestamp"] < segment.touchdown_time]
    final_30 = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=30)].copy()
    final_10 = last_airborne[last_airborne["Timestamp"] >= segment.touchdown_time - pd.Timedelta(seconds=10)].copy()

    for frame in (final_30, final_10, touchdown):
        for col in ["ROLL_C", "PITCH_C", "VRTG_C", "IVV_C", "N11_C", "N12_C", "TORQ1", "TORQ2", "NP1", "NP2", "IAS_C"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    bank_max = float(final_30["ROLL_C"].abs().max()) if not final_30.empty else float("nan")
    bank_touchdown = float(touchdown["ROLL_C"].iloc[0]) if not touchdown.empty else float("nan")
    pitch_max = float(final_30["PITCH_C"].max()) if not final_30.empty else float("nan")
    pitch_touchdown = float(touchdown["PITCH_C"].iloc[0]) if not touchdown.empty else float("nan")
    g_peak = float(final_10["VRTG_C"].max()) if not final_10.empty else float("nan")
    g_touchdown = float(touchdown["VRTG_C"].iloc[0]) if not touchdown.empty else float("nan")
    sink_peak = float((-final_30["IVV_C"]).max()) if not final_30.empty else float("nan")

    left_col = get_numeric_series(final_30, ["N11_C", "TORQ1", "NP1"])
    right_col = get_numeric_series(final_30, ["N12_C", "TORQ2", "NP2"])
    left_touch = get_numeric_series(final_10, ["N11_C", "TORQ1", "NP1"])
    right_touch = get_numeric_series(final_10, ["N12_C", "TORQ2", "NP2"])

    n1_pre = float(left_col.head(max(1, len(final_30) // 2)).mean()) if left_col is not None and not final_30.empty else float("nan")
    n1_touch = float(left_touch.mean()) if left_touch is not None and not final_10.empty else float("nan")
    n2_pre = float(right_col.head(max(1, len(final_30) // 2)).mean()) if right_col is not None and not final_30.empty else float("nan")
    n2_touch = float(right_touch.mean()) if right_touch is not None and not final_10.empty else float("nan")
    thrust_proxy_delta = None
    thrust_state = "n/a"
    if not pd.isna(n1_pre) and not pd.isna(n1_touch) and not pd.isna(n2_pre) and not pd.isna(n2_touch):
        thrust_proxy_delta = ((n1_touch + n2_touch) / 2) - ((n1_pre + n2_pre) / 2)
        if thrust_proxy_delta <= -0.5:
            thrust_state = "thrust easing toward idle"
        elif thrust_proxy_delta >= 0.5:
            thrust_state = "thrust increasing"
        else:
            thrust_state = "thrust broadly steady"

    notes = []
    if not pd.isna(bank_max):
        if bank_max <= 5:
            notes.append("Bank stayed modest in the last 30 seconds.")
        else:
            notes.append("Bank excursion was noticeable and should be watched in the flare.")
    if not pd.isna(pitch_max):
        if pitch_max <= 6:
            notes.append("Pitch remained within a normal landing envelope.")
        else:
            notes.append("Pitch increased more aggressively than a typical landing flare.")
    if not pd.isna(g_peak):
        if g_peak <= 1.2:
            notes.append("Vertical acceleration stayed soft and well damped.")
        elif g_peak <= 1.4:
            notes.append("Vertical acceleration was firmer but still inside a normal landing range.")
        else:
            notes.append("Vertical acceleration was high enough to suggest a firm touchdown.")
    if not pd.isna(sink_peak):
        if sink_peak <= 700:
            notes.append("Sink rate was well controlled.")
        elif sink_peak <= 1000:
            notes.append("Sink rate stayed within the stabilized-approach limit but deserves attention.")
        else:
            notes.append("Sink rate exceeded the stabilized-approach target.")
    if thrust_state != "n/a":
        notes.append(f"Thrust proxy from available engine columns suggests {thrust_state}; actual thrust lever angle is not recorded.")

    return {
        "metrics": [
            {"label": "Max bank", "value": f"{bank_max:.1f} deg" if not pd.isna(bank_max) else "n/a"},
            {"label": "Touchdown bank", "value": f"{bank_touchdown:.1f} deg" if not pd.isna(bank_touchdown) else "n/a"},
            {"label": "Max pitch", "value": f"{pitch_max:.1f} deg" if not pd.isna(pitch_max) else "n/a"},
            {"label": "Touchdown pitch", "value": f"{pitch_touchdown:.1f} deg" if not pd.isna(pitch_touchdown) else "n/a"},
            {"label": "Peak vertical accel", "value": f"{g_peak:.2f} g" if not pd.isna(g_peak) else "n/a"},
            {"label": "Touchdown g", "value": f"{g_touchdown:.2f} g" if not pd.isna(g_touchdown) else "n/a"},
            {"label": "Peak sink", "value": f"{sink_peak:.0f} fpm" if not pd.isna(sink_peak) else "n/a"},
            {"label": "Thrust proxy", "value": thrust_state},
        ],
        "notes": notes,
        "thrust_proxy_delta": thrust_proxy_delta,
    }


def aircraft_trace_list(origin: tuple[float, float, float], pitch_deg: float, roll_deg: float, yaw_deg: float, opacity: float, suffix: str = "") -> list[go.Mesh3d]:
    return build_aircraft_model(
        origin=origin,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        yaw_deg=yaw_deg,
        opacity=opacity,
        name_suffix=suffix,
    )


def analyze_file(path: Path | str) -> dict:
    path = Path(path)
    df = normalize_dataframe(read_dataframe_from_path(path))
    segment = segment_landing(df)
    briefing = build_rule_assessment(segment)
    summary_fig = build_summary_figure(segment)
    figure_3d = build_3d_figure(segment)
    replay_3d = build_runway_replay_figure(segment)

    return {
        "filename": path.name,
        "rows": len(df),
        "touchdown_time": segment.touchdown_time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_fig": summary_fig,
        "figure_3d": figure_3d,
        "replay_3d": replay_3d,
        "briefing": briefing,
        "control_technique": build_control_technique(segment),
        "replay_metrics": build_replay_metrics(segment),
        "rule_catalog": [
            {"title": rule.title, "statement": rule.statement, "source": rule.source}
            for rule in LANDING_RULES
        ],
        "pdf_available": REPORTLAB_AVAILABLE,
        "landing_rows": len(segment.landing),
        "approach_rows": len(segment.approach),
        "flare_rows": len(segment.flare),
        "touchdown_rows": len(segment.touchdown),
    }


def build_pdf_report(result: dict) -> io.BytesIO:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF export requires reportlab, which is not installed.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=42,
        bottomMargin=36,
        title=f"{Path(result['filename']).stem} debrief",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11, alignment=TA_LEFT, spaceAfter=4))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT, textColor=colors.HexColor("#475569")))

    story = []
    story.append(Paragraph("Landing Debrief Report", styles["Title"]))
    story.append(Paragraph(f"File: {result['filename']}", styles["BodyText"]))
    story.append(Paragraph(f"Touchdown time: {result['touchdown_time']}", styles["BodyText"]))
    story.append(Paragraph(f"Overall assessment: {result['briefing']['overall']}  Score: {result['briefing']['score']}/100", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Control Technique", styles["Heading2"]))
    control_rows = [["Metric", "Value"]]
    for metric in result["control_technique"]["metrics"]:
        control_rows.append([Paragraph(metric["label"], styles["Small"]), Paragraph(metric["value"], styles["Small"])])
    control_table = Table(control_rows, colWidths=[220, 250], repeatRows=1)
    control_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(control_table)
    story.append(Spacer(1, 8))
    for note in result["control_technique"]["notes"]:
        story.append(Paragraph(f"• {note}", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Rule Checks", styles["Heading2"]))
    rule_rows = [[Paragraph("Rule", styles["Small"]), Paragraph("Status", styles["Small"]), Paragraph("Source", styles["Small"]), Paragraph("Assessment", styles["Small"])]]
    for rule in result["briefing"]["rule_checks"]:
        rule_rows.append([
            Paragraph(rule["title"], styles["Small"]),
            Paragraph(rule["status"], styles["Small"]),
            Paragraph(rule["source"], styles["Tiny"]),
            Paragraph(rule["rationale"], styles["Small"]),
        ])
    rule_table = Table(rule_rows, colWidths=[115, 50, 165, 160], repeatRows=1)
    rule_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(rule_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Observations", styles["Heading2"]))
    for item in result["briefing"]["findings"]:
        story.append(Paragraph(f"• {item}", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Improvement Areas", styles["Heading2"]))
    for item in result["briefing"]["improvements"]:
        story.append(Paragraph(f"• {item}", styles["Small"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Extracted Rules", styles["Heading2"]))
    rule_catalog_rows = [[Paragraph("Rule", styles["Small"]), Paragraph("Source", styles["Small"]), Paragraph("Statement", styles["Small"])]]
    for rule in result["rule_catalog"]:
        rule_catalog_rows.append([
            Paragraph(rule["title"], styles["Small"]),
            Paragraph(rule["source"], styles["Tiny"]),
            Paragraph(rule["statement"], styles["Small"]),
        ])
    rule_catalog_table = Table(rule_catalog_rows, colWidths=[125, 145, 205], repeatRows=1)
    rule_catalog_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(rule_catalog_table)

    def draw_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawRightString(doc_obj.pagesize[0] - 36, 20, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    buffer.seek(0)
    return buffer


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            error = "Choose a CSV or ZIP file first."
        elif not allowed_file(uploaded.filename):
            error = "Only CSV and ZIP files are supported."
        else:
            filename = secure_filename(uploaded.filename)
            destination = UPLOAD_DIR / filename
            uploaded.save(destination)
            try:
                result = analyze_file(destination)
            except Exception as exc:
                error = str(exc)

    return render_template("index.html", result=result, error=error)


@app.route("/download-report/<path:filename>")
def download_report(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        abort(404)
    if not REPORTLAB_AVAILABLE:
        abort(503, description="PDF export is unavailable because reportlab is not installed.")

    result = analyze_file(path)
    pdf_buffer = build_pdf_report(result)
    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{Path(filename).stem}_debrief.pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
