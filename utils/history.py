"""Verification-history persistence backed by the application database."""

import pandas as pd

from db.repository import create_verification_record, list_verification_records


HISTORY_COLUMNS = ["date", "person", "result", "distance", "threshold", "engine"]


def save_verification_history(
    person_name,
    result,
    distance,
    threshold,
    person_id=None,
    engine=None,
):
    return create_verification_record(
        person_name=person_name,
        result=result,
        distance=float(distance),
        threshold=float(threshold) if threshold is not None else None,
        person_id=person_id,
        engine=engine,
    )


def load_verification_history():
    records = list_verification_records()
    if not records:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    frame = pd.DataFrame(records)
    return frame[[column for column in HISTORY_COLUMNS if column in frame.columns]]
