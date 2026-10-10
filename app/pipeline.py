"""Data pipeline: cleaning, features, training and prediction.

The logic is ported from notebook v4 (same rules, same features, same split).
"""
import difflib
import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ===== Parameters (same defaults as the notebook) =====
DUE_DAYS = 10             # a payment is late if made > DUE_DAYS after class 1
PRE_PAYMENT_WINDOW = 7    # payments up to 7 days before class 1 count for that cycle
ABSENCE_STATES = ["Falta / No asistió", "Licencia"]
CLASS_STATES = ["Asistió / Clase realizada", "Falta / No asistió", "Licencia"]
TEST_FRACTION = 0.25      # last 25% of the timeline is used only for testing
MIN_ROWS = 30             # below this the dataset is too small to train
SHORT_HISTORY = 5         # fewer classes than this -> low-confidence warning
STALE_DAYS = 90           # requested date this far after the last record -> warning

REQUIRED_COLUMNS = ["Estudiante", "Materia / Especialidad", "Fecha", "Nro_Clase",
                    "Estado_Asistencia", "Detalle_Pago_Observaciones"]
LATE_COLS = ["past_late_fees", "late_fee_rate", "overdue_now", "days_since_payment", "n_payments"]
ABSENCE_COLS = ["past_no_shows", "past_absences", "absence_rate", "absences_last5", "n_classes"]


class ApiError(Exception):
    """Error with an HTTP status and a user-facing message (in Spanish)."""

    def __init__(self, status_code, message, suggestions=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.suggestions = suggestions or []


# ===== Baseline rules (scores in 0..1, used only to compare against the model) =====
def late_rule(d):
    return (d["overdue_now"] * 2 + d["past_late_fees"]).clip(upper=5) / 5


def absence_rule(d):
    return (d["past_no_shows"] * 2 + d["absences_last5"] + d["absence_rate"]).clip(upper=5) / 5


EVENTS = {
    "late_payment": ("y_late", LATE_COLS, late_rule),
    "absence": ("y_absence", ABSENCE_COLS, absence_rule),
}


@dataclass
class Bundle:
    """Everything the API needs after training."""
    data: pd.DataFrame
    models: dict
    metrics: list
    split_date: pd.Timestamp
    dropped_rows: int
    last_record: pd.Timestamp


# ===== 1. Cleaning =====
def read_csv_text(text):
    if not text.strip():
        raise ApiError(400, "El archivo está vacío.")
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        raise ApiError(400, "No se pudo leer el archivo como CSV.")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ApiError(400, "Faltan columnas en el CSV: " + ", ".join(missing))
    return raw


def clean(raw):
    df = raw.rename(columns={"Materia / Especialidad": "subject",
                             "Detalle_Pago_Observaciones": "detail",
                             "Estudiante": "student", "Fecha": "date",
                             "Estado_Asistencia": "status"}).copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    dropped = int(df["date"].isna().sum())
    df = df.dropna(subset=["date", "student"])
    df["student"] = df["student"].astype(str).str.strip()
    df["detail"] = df["detail"].fillna("").astype(str)
    df["class_number"] = pd.to_numeric(df["Nro_Clase"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    # a payment = detail mentions "mens"/"saldo" or an amount in bs (except textbook purchases)
    df["paid"] = (df["detail"].str.contains(r"mens|saldo|\d+\s*bs", case=False, regex=True)
                  & ~df["detail"].str.contains("texto", case=False))
    df = df.sort_values(["student", "date"]).reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise ApiError(400, f"Datos insuficientes: se necesitan al menos {MIN_ROWS} filas válidas y hay {len(df)}.")
    return df, dropped


# ===== 2. Fees (cycles) =====
def build_fees(df):
    """A cycle starts at 'clase 1' and is one fee. Late = paid after the due date or never paid."""
    df["cycle"] = (df["class_number"] == 1).groupby(df["student"]).cumsum()
    rows = []
    for student, g in df.groupby("student"):
        payments = g.loc[g["paid"], "date"].tolist()
        starts = g[g["class_number"] == 1].groupby("cycle")["date"].min()
        for cycle, start in starts.items():
            candidates = [p for p in payments if p >= start - pd.Timedelta(days=PRE_PAYMENT_WINDOW)]
            following = starts[starts.index > cycle]
            limit = following.iloc[0] - pd.Timedelta(days=PRE_PAYMENT_WINDOW) if len(following) else None
            candidates = [p for p in candidates if limit is None or p < limit]
            rows.append({"student": student, "cycle": cycle, "start": start,
                         "payment_date": min(candidates) if candidates else pd.NaT})
    fees = pd.DataFrame(rows, columns=["student", "cycle", "start", "payment_date"])
    fees["due"] = fees["start"] + pd.Timedelta(days=DUE_DAYS)
    last_date = df["date"].max()
    # unknown (NaN) when there is no payment yet and the due date has not passed
    fees["late"] = np.where(fees["payment_date"].notna(),
                            (fees["payment_date"] > fees["due"]).astype(float),
                            np.where(fees["due"] < last_date, 1.0, np.nan))
    return fees


# ===== 3. Features per (student, cutoff date): only history <= cutoff =====
def build_features(df, fees):
    classes = df[df["status"].isin(CLASS_STATES)].copy()
    classes["absent"] = classes["status"].isin(ABSENCE_STATES).astype(int)
    rows = []
    for student, g in classes.groupby("student"):
        g = g.sort_values("date").reset_index(drop=True)
        student_fees = fees[fees["student"] == student].sort_values("start")
        student_payments = df[(df["student"] == student) & df["paid"]]
        for i in range(len(g)):
            cutoff = g.loc[i, "date"]
            hist = g.iloc[:i + 1]
            last5 = hist.tail(5)
            payments_so_far = student_payments[student_payments["date"] <= cutoff]
            past_fees = student_fees[student_fees["start"] <= cutoff]
            known = past_fees[(past_fees["payment_date"] <= cutoff) | (past_fees["due"] < cutoff)]
            late = known[known["payment_date"].isna() | (known["payment_date"] > known["due"])]
            current = past_fees.iloc[-1] if len(past_fees) else None
            overdue_now = int(current is not None and current["due"] < cutoff and
                              (pd.isna(current["payment_date"]) or current["payment_date"] > cutoff))
            days_since = (cutoff - payments_so_far["date"].max()).days if len(payments_so_far) else 999
            # targets: only for training/testing, never used as features
            next_class = g.iloc[i + 1] if i + 1 < len(g) else None
            next_fee = student_fees[student_fees["start"] > cutoff].head(1)
            rows.append({
                "student": student, "cutoff": cutoff,
                "past_late_fees": len(late),
                "late_fee_rate": len(late) / len(known) if len(known) else 0,
                "overdue_now": overdue_now,
                "days_since_payment": min(days_since, 120),
                "n_payments": len(payments_so_far),
                "past_no_shows": int((hist["status"] == "Falta / No asistió").sum()),
                "past_absences": int(hist["absent"].sum()),
                "absence_rate": hist["absent"].mean(),
                "absences_last5": int(last5["absent"].sum()),
                "n_classes": len(hist),
                "y_absence": np.nan if next_class is None else float(next_class["absent"]),
                "y_late": float(next_fee["late"].iloc[0]) if len(next_fee) else np.nan,
            })
    return pd.DataFrame(rows)


# ===== 4. Temporal split + training + metrics =====
def split_sets(data, fees, y_col, split_date):
    d = data.dropna(subset=[y_col])
    train = d[d["cutoff"] < split_date]
    test = d[d["cutoff"] >= split_date]
    if y_col == "y_late" and len(train):
        # a training fee must be due before the split date
        next_due = train.apply(
            lambda r: fees[(fees.student == r.student) & (fees.start > r.cutoff)]["due"].min(), axis=1)
        train = train[next_due < split_date]
    return train, test


def train(text):
    """Full pipeline: CSV text -> trained Bundle."""
    raw = read_csv_text(text)
    df, dropped = clean(raw)
    fees = build_fees(df)
    data = build_features(df, fees)
    cutoffs = data["cutoff"].sort_values()
    split_date = cutoffs.iloc[int(len(cutoffs) * (1 - TEST_FRACTION))]

    models, metrics = {}, []
    for event, (y_col, cols, rule) in EVENTS.items():
        tr, te = split_sets(data, fees, y_col, split_date)
        if len(tr) < 10 or tr[y_col].nunique() < 2:
            raise ApiError(400, "Datos insuficientes para entrenar: faltan ejemplos de mora o de inasistencia "
                                "en el periodo de entrenamiento.")
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
        model.fit(tr[cols], tr[y_col])
        models[event] = model
        can_score = len(te) > 0 and te[y_col].nunique() > 1
        metrics.append({
            "event": event,
            "test_prevalence": float(te[y_col].mean()) if len(te) else None,
            "ap_baseline": float(average_precision_score(te[y_col], rule(te))) if can_score else None,
            "ap_model": float(average_precision_score(te[y_col], model.predict_proba(te[cols])[:, 1]))
            if can_score else None,
        })
    return Bundle(data=data, models=models, metrics=metrics, split_date=split_date,
                  dropped_rows=dropped, last_record=df["date"].max())


# ===== 5. Prediction with validation =====
def parse_date(text):
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(str(text).strip(), format=fmt)
        except (ValueError, TypeError):
            continue
    raise ApiError(422, f"La fecha '{text}' no es válida. Usa el formato dd/mm/aaaa.")


def predict(bundle, student, date_text):
    student = (student or "").strip()
    if not student:
        raise ApiError(422, "El nombre del estudiante está vacío.")
    names = bundle.data["student"].unique().tolist()
    match = [n for n in names if n.lower() == student.lower()]
    if not match:
        close = difflib.get_close_matches(student, names, n=3, cutoff=0.6)
        raise ApiError(404, f"No existe el estudiante '{student}'.", close)
    date = parse_date(date_text)
    history = bundle.data[bundle.data["student"] == match[0]]
    if date < history["cutoff"].min():
        raise ApiError(422, f"{match[0]} no tiene historial antes del {history['cutoff'].min().date()}.")

    row = history[history["cutoff"] <= date].iloc[[-1]]   # last cutoff available <= date
    probs = {event: float(bundle.models[event].predict_proba(row[cols])[0, 1])
             for event, (_, cols, _) in EVENTS.items()}

    warnings = []
    n_classes, n_payments = int(row["n_classes"].iloc[0]), int(row["n_payments"].iloc[0])
    if n_classes < SHORT_HISTORY:
        warnings.append(f"Solo hay {n_classes} clases registradas; el estimado es poco confiable.")
    if n_payments == 0:
        warnings.append("No hay ningún pago registrado; el porcentaje de mora puede reflejar solo falta de información.")
    gap = (date - row["cutoff"].iloc[0]).days
    if gap > STALE_DAYS:
        warnings.append(f"El último registro es del {row['cutoff'].iloc[0].date()}; "
                        "la predicción no incluye lo ocurrido después.")
    return {
        "student": match[0],
        "requested_date": str(date.date()),
        "cutoff_used": str(row["cutoff"].iloc[0].date()),
        "probabilities": probs,
        "base_rates": {"late_payment": float(bundle.data["y_late"].mean()),
                       "absence": float(bundle.data["y_absence"].mean())},
        "history": {"classes": n_classes, "payments": n_payments,
                    "absences": int(row["past_absences"].iloc[0])},
        "warnings": warnings,
    }


def summary(bundle):
    """Information shown by the frontend after training."""
    return {
        "students": sorted(bundle.data["student"].unique().tolist()),
        "n_examples": int(len(bundle.data)),
        "split_date": str(bundle.split_date.date()),
        "last_record": str(bundle.last_record.date()),
        "dropped_rows": bundle.dropped_rows,
        "metrics": bundle.metrics,
    }
