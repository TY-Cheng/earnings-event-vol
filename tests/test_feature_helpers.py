from __future__ import annotations

import pandas as pd
import pytest

from earnings_event_vol import features


def test_numeric_and_timestamp_helpers_fill_missing_columns() -> None:
    frame = pd.DataFrame(index=[10, 11])

    numeric = features._numeric_or_nan(frame, "missing_numeric")
    timestamp = features._timestamp_series(frame, "missing_timestamp")

    assert numeric.index.tolist() == [10, 11]
    assert numeric.isna().all()
    assert str(numeric.dtype) == "Float64"
    assert timestamp.index.tolist() == [10, 11]
    assert timestamp.isna().all()
    assert str(timestamp.dtype) == "datetime64[ns, UTC]"


def test_entry_timestamp_series_accepts_explicit_timestamp() -> None:
    frame = pd.DataFrame({"event_entry_timestamp": ["2026-06-05T19:45:00Z"]})

    out = features._entry_timestamp_series(frame)

    assert out.iloc[0] == pd.Timestamp("2026-06-05T19:45:00Z")


def test_entry_timestamp_series_requires_entry_information() -> None:
    with pytest.raises(ValueError, match="event_entry_timestamp or entry_date"):
        features._entry_timestamp_series(pd.DataFrame({"ticker": ["AAPL"]}))


def test_entry_timestamp_series_rejects_missing_entry_values() -> None:
    with pytest.raises(ValueError, match="missing event_entry_timestamp"):
        features._entry_timestamp_series(pd.DataFrame({"entry_date": [pd.NA]}))
