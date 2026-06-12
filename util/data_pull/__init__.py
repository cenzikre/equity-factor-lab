"""FMP bulk data pull: universe -> raw per-component parquet files on S3.

Submodules:
- components: data component registry and request-spec builders
- pull: pull orchestration (fetch, normalize, write dated S3 snapshot)

Pipeline position:
    build_stock_universe -> data pull (this package) -> util.dataset_builder
"""
