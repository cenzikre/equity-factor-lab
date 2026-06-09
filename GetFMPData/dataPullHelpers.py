import pandas as pd
import numpy as np
import warnings


def normalize_df_for_parquet(
    df: pd.DataFrame,
    *,
    datetime_success_ratio: float = 0.85,
    numeric_success_ratio: float = 0.90,
    treat_huge_int_as_string: bool = True,
    huge_int_threshold: int = 2**53 - 1,  # max exact integer in float64
) -> pd.DataFrame:
    """
    Normalize dtypes for mixed-schema API data so pyarrow/parquet writes won't fail.

    Strategy for object columns:
      - If values mostly parse as numeric -> numeric (Float64)
      - Else if values mostly parse as datetime (STRING-LIKE only) -> datetime64
      - Else -> pandas string dtype

    Key fix vs prior version:
      - Prevent numeric object columns from being parsed as datetimes.
    """
    df = df.copy()

    for col in df.columns:
        s = df[col]

        # Skip already-good dtypes
        if (
            pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_numeric_dtype(s)
            or pd.api.types.is_bool_dtype(s)
        ):
            continue

        # Only attempt inference on object/string-like columns
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        non_null = s.dropna()
        if non_null.empty:
            df[col] = s.astype("string")
            continue

        if col in ['cik']:
            df[col] = s.astype("string")
            continue

        # --- 1) Try numeric FIRST (fast, avoids numeric->datetime traps) ---
        num = pd.to_numeric(non_null, errors="coerce")
        num_ratio = num.notna().mean()

        if num_ratio >= numeric_success_ratio:
            if treat_huge_int_as_string:
                full_num = pd.to_numeric(s, errors="coerce")
                huge_mask = s.notna() & full_num.notna() & (full_num.abs() > huge_int_threshold)

                if huge_mask.any():
                    # Preserve huge values as strings; numeric elsewhere
                    out = pd.Series([pd.NA] * len(s), index=s.index, dtype="string")
                    out[~huge_mask] = pd.to_numeric(s[~huge_mask], errors="coerce").astype("float64").astype("string")
                    out[huge_mask] = s[huge_mask].astype("string")
                    df[col] = out
                else:
                    df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            else:
                df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            continue

        # --- 2) Try datetime ONLY if values are string-like ---
        # Convert to string for checking, but don't force datetime parsing on numeric objects.
        sample_str = non_null.astype(str)

        # Heuristic: only attempt datetime if a good fraction looks like dates/times
        # (has '-' or '/' or ':' or 'T' like ISO8601)
        looks_date = sample_str.str.contains(r"[-/:T]", regex=True, na=False).mean()

        if looks_date >= 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dt = pd.to_datetime(non_null, errors="coerce", utc=False, cache=False)

            dt_ratio = dt.notna().mean()
            if dt_ratio >= datetime_success_ratio:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    df[col] = pd.to_datetime(s, errors="coerce", utc=False, cache=False)
                continue

        # --- 3) Fallback: keep string ---
        df[col] = s.astype("string")

    return df
