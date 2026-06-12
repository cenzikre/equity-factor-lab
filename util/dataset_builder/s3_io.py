"""S3 parquet I/O helpers for the dataset build.

Raw per-datatype files follow the naming convention
    {prefix}/data_{datatype}_tk{T}_pd{P}.parquet
and a datatype may be split across several period batches (pd1, pd2, ...),
which are concatenated on load.
"""

import re
from typing import Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .column_specs import S3_BUCKET, S3_PREFIX, S3_REGION


def get_s3_filesystem(region: str = S3_REGION) -> pafs.S3FileSystem:
    return pafs.S3FileSystem(region=region)


def dataset_base_path(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX) -> str:
    """Bucket-relative base path as expected by pyarrow's S3FileSystem."""
    return f"{bucket}/{prefix}"


def raw_snapshot_path(label: str, bucket: str = S3_BUCKET,
                      prefix: str = S3_PREFIX) -> str:
    """Base path of one dated raw-pull snapshot, e.g. .../raw/20260611.

    Raw component files written by `util.data_pull` live under per-date
    directories; legacy (pre-snapshot) files sit at the dataset root.
    """
    return f"{bucket}/{prefix}/raw/{label}"


def write_parquet_df(fs: pafs.FileSystem, df: pd.DataFrame, path: str,
                     compression: str = "zstd") -> None:
    """Write a DataFrame as one parquet file on the given filesystem."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, filesystem=fs, compression=compression)


def write_json(fs: pafs.FileSystem, obj, path: str) -> None:
    import json
    with fs.open_output_stream(path) as f:
        f.write(json.dumps(obj, indent=2, default=str).encode("utf-8"))


def list_datatype_files(fs: pafs.S3FileSystem, datatype: str,
                        base_path: Optional[str] = None) -> List[str]:
    """All raw parquet files for one datatype, ordered by period batch."""
    base_path = base_path or dataset_base_path()
    infos = fs.get_file_info(pafs.FileSelector(base_path))
    pattern = re.compile(rf"data_{re.escape(datatype)}_tk\d+_pd(\d+)\.parquet$")
    matches = []
    for info in infos:
        m = pattern.search(info.path)
        if m:
            matches.append((int(m.group(1)), info.path))
    if not matches:
        raise FileNotFoundError(f"no raw files for datatype '{datatype}' under {base_path}")
    return [path for _, path in sorted(matches)]


def load_datatype(fs: pafs.S3FileSystem, datatype: str,
                  columns: Optional[List[str]] = None,
                  symbols: Optional[Iterable[str]] = None,
                  base_path: Optional[str] = None) -> pd.DataFrame:
    """Load one datatype (all period batches concatenated) as a DataFrame.

    `symbols` enables predicate-pushdown filtering so only the requested
    ticker batch is materialized — this is what keeps the build inside the
    kernel's memory budget.
    """
    files = list_datatype_files(fs, datatype, base_path)
    dset = ds.dataset(files, filesystem=fs, format="parquet")
    filt = ds.field("symbol").isin(list(symbols)) if symbols is not None else None
    table = dset.to_table(columns=columns, filter=filt)
    return table.to_pandas()


def load_single_parquet(fs: pafs.S3FileSystem, filename: str,
                        columns: Optional[List[str]] = None,
                        base_path: Optional[str] = None) -> pd.DataFrame:
    base_path = base_path or dataset_base_path()
    table = pq.read_table(fs.open_input_file(f"{base_path}/{filename}"), columns=columns)
    return table.to_pandas()


def symbol_row_counts(fs: pafs.S3FileSystem, datatypes: Iterable[str],
                      base_path: Optional[str] = None) -> pd.Series:
    """Daily row count per symbol across the given price datatypes.

    Used to balance ticker batches so each batch materializes a similar
    number of panel rows.
    """
    counts: Optional[pd.Series] = None
    for datatype in datatypes:
        files = list_datatype_files(fs, datatype, base_path)
        dset = ds.dataset(files, filesystem=fs, format="parquet")
        col = dset.to_table(columns=["symbol"]).column("symbol")
        vc = col.value_counts().to_pandas()
        s = pd.Series(
            [v["counts"] for v in vc], index=[v["values"] for v in vc], dtype="int64"
        )
        counts = s if counts is None else counts.add(s, fill_value=0)
    return counts.sort_values(ascending=False).astype("int64")


def make_symbol_batches(row_counts: pd.Series, n_batches: int) -> List[List[str]]:
    """Greedy bin-packing of symbols into n_batches of ~equal total rows."""
    totals = [0] * n_batches
    batches: List[List[str]] = [[] for _ in range(n_batches)]
    for sym, n in row_counts.items():  # descending count order
        i = totals.index(min(totals))
        batches[i].append(sym)
        totals[i] += int(n)
    return [b for b in batches if b]


class StreamingParquetWriter:
    """Append pandas batches into a single parquet file on S3.

    The schema is locked from the first batch; later batches are converted
    with the same schema so the row groups stay consistent.
    """

    def __init__(self, fs: pafs.S3FileSystem, path: str, compression: str = "zstd"):
        self.fs = fs
        self.path = path
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None
        self.schema: Optional[pa.Schema] = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self._writer is None:
            self.schema = table.schema
            self._writer = pq.ParquetWriter(
                self.path, table.schema, filesystem=self.fs, compression=self.compression
            )
        self._writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
