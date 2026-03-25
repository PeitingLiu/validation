import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(r"D:\athesis-data\combine-patien-basic-information.xlsx")
OUTPUT_PATH = Path(r"D:\athesis-data\combine-patien-basic-information.cleaned.xlsx")


STATIC_FIRST = [
    "sample-id",
    "name",
    "group",
    "origin-id",
    "Age, years",
    "Female, n (%)",
    "hospital stay",
]

NUMERIC_MAX = [
    "Procalcitonin [ng/mL]",
    "C-reactive protein [mg/dL]",
    "IL-6",
    "WBC [x10^3/L]",
    "neutrophil[x10^3/L]",
    "SOFA",
    "APACHE II",
    "PT",
    "APTT",
    "TT",
    "DD",
    "Lactate, mmol/L",
    "Total bilirubin, mg/dL",
    "Blood urea nitrogen (BUN), mg/dL",
    "Heart rate, bpm",
    "respiratory rate",
    "WBC [x10^3/L].1",
    "respiratory rate.1",
    "Blood urea nitrogen (BUN), mg/dL.1",
    "Creatinine, mg/dL",
    "Total bilirubin, mg/dL.1",
    "Alanine transaminas （ALT）[U/L]",
    "Aspartate transaminase (AST), U/L",
    "TNT",
    "LDH",
    "CK-MB",
]

NUMERIC_MIN = [
    "FIB",
    "Platelet count, ×10⁹/L",
    "Urine output, mL/day",
    "PaCO2",
    "PaO2/FiO2",
    "Urine output, mL/day.1",
]

BINARY_ANY = {
    "shock": ["是", "休克"],
    "INR": ["是"],
    "AKI": ["是", "肾损伤"],
}


def clean_missing(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "/", " /", "nan", "None", "N/A", "NA", "？"}:
            return pd.NA
        return stripped
    return value


def first_non_null(series):
    cleaned = [clean_missing(v) for v in series]
    for value in cleaned:
        if pd.notna(value):
            return value
    return pd.NA


def parse_numeric(value):
    value = clean_missing(value)
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def aggregate_numeric(series, mode):
    numbers = [parse_numeric(v) for v in series]
    numbers = [v for v in numbers if v is not None]
    if not numbers:
        return pd.NA
    return max(numbers) if mode == "max" else min(numbers)


def aggregate_binary(series, positive_tokens):
    values = [clean_missing(v) for v in series]
    texts = [str(v) for v in values if pd.notna(v)]
    for text in texts:
        if any(token in text for token in positive_tokens):
            return text
    for text in texts:
        if "否" in text:
            return text
    return pd.NA


def ards_score(text):
    if pd.isna(text):
        return -1
    text = str(text)
    if "重度" in text:
        return 3
    if "中度" in text:
        return 2
    if "轻度" in text:
        return 1
    if "否" in text:
        return 0
    return -1


def aggregate_ards(series):
    values = [clean_missing(v) for v in series]
    best_value = pd.NA
    best_score = -1
    for value in values:
        score = ards_score(value)
        if score > best_score:
            best_score = score
            best_value = value
    return best_value


def mech_vent_score(text):
    if pd.isna(text):
        return -1
    text = str(text)
    if any(token in text for token in ["气管切开", "气管插管", "经口插管", "经鼻气管插管", "呼吸机"]):
        return 3
    if any(token in text for token in ["无创", "面罩", "鼻导管"]):
        return 2
    if "是" in text:
        return 1
    if "否" in text:
        return 0
    return -1


def aggregate_mech_vent(series):
    values = [clean_missing(v) for v in series]
    best_value = pd.NA
    best_score = -1
    for value in values:
        score = mech_vent_score(value)
        if score > best_score:
            best_score = score
            best_value = value
    return best_value


def pao2fio2_score(text):
    if pd.isna(text):
        return None
    text = str(text)
    mapping = {
        "氧合尚可": 2,
        "氧合一般": 1,
    }
    if text in mapping:
        return mapping[text]
    return parse_numeric(text)


def aggregate_pao2fio2(series):
    scored = []
    for value in series:
        cleaned = clean_missing(value)
        score = pao2fio2_score(cleaned)
        if score is not None:
            scored.append((score, cleaned))
    if not scored:
        return pd.NA
    return min(scored, key=lambda item: item[0])[1]


def normalize_hospital_stay(value):
    number = parse_numeric(value)
    return int(number) if number is not None else pd.NA


def build_rules(columns):
    rules = []
    for column in columns:
        if column in STATIC_FIRST:
            rule = "首次非空值"
        elif column in NUMERIC_MAX:
            rule = "按严重程度取最大值"
        elif column in NUMERIC_MIN:
            rule = "按严重程度取最小值"
        elif column in BINARY_ANY:
            rule = "任一阳性即阳性"
        elif column == "ARDS":
            rule = "按 否<轻度<中度<重度 取最重"
        elif column == "Mechanical ventilation, n (%)":
            rule = "按 无<低流量/无创<插管/呼吸机 取最重"
        else:
            rule = "首次非空值"
        rules.append({"column": column, "rule": rule})
    return pd.DataFrame(rules)


def main():
    df = pd.read_excel(INPUT_PATH)
    if not df.empty and str(df.iloc[0]["sample-id"]).strip() == "患者者编号":
        df = df.iloc[1:].copy()
    df = df.apply(lambda col: col.map(clean_missing))
    df["sample-id"] = df["sample-id"].ffill()
    df["origin-id"] = df["origin-id"].ffill()
    df["name"] = df["name"].ffill()

    patient_key = df["origin-id"].where(df["origin-id"].notna(), df["sample-id"])
    df["_patient_key"] = patient_key

    cleaned_rows = []
    for _, group in df.groupby("_patient_key", sort=False):
        row = {}
        for column in df.columns:
            if column == "_patient_key":
                continue
            if column == "hospital stay":
                row[column] = normalize_hospital_stay(first_non_null(group[column]))
            elif column in STATIC_FIRST:
                row[column] = first_non_null(group[column])
            elif column in NUMERIC_MAX:
                row[column] = aggregate_numeric(group[column], "max")
            elif column in NUMERIC_MIN:
                if column == "PaO2/FiO2":
                    row[column] = aggregate_pao2fio2(group[column])
                else:
                    row[column] = aggregate_numeric(group[column], "min")
            elif column in BINARY_ANY:
                row[column] = aggregate_binary(group[column], BINARY_ANY[column])
            elif column == "ARDS":
                row[column] = aggregate_ards(group[column])
            elif column == "Mechanical ventilation, n (%)":
                row[column] = aggregate_mech_vent(group[column])
            else:
                row[column] = first_non_null(group[column])
        row["merged_rows"] = len(group)
        cleaned_rows.append(row)

    cleaned_df = pd.DataFrame(cleaned_rows)
    rules_df = build_rules([c for c in cleaned_df.columns if c != "merged_rows"])

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        cleaned_df.to_excel(writer, index=False, sheet_name="cleaned")
        rules_df.to_excel(writer, index=False, sheet_name="rules")

    print(f"cleaned_rows={len(cleaned_df)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
