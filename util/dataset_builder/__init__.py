"""Modular helpers for constructing the full daily fundamental panel dataset.

Submodules:
- column_specs: per-table column selections, rename maps, merge step specs
- s3_io: parquet read/write helpers against the S3 dataset bucket
- merge_core: point-in-time merge logic, liquidity flags, build statistics
- report: markdown summary report rendering
"""
