import polars as pl

from computer_vision.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def interim():
    files = list(RAW_DATA_DIR.glob("*.csv.tar.gz"))
    files.sort()
    for file in files:
        print(f"Processing {file}...")
        lf = pl.scan_csv(
            file,
            new_columns=["datetime", "id", "value"],
            ignore_errors=True,
            schema={"datetime": pl.Int32, "id": pl.Int32, "value": pl.Float32},
            low_memory=True,
        )
        lf = lf.drop_nulls("datetime")
        lf = lf.with_columns(
            pl.from_epoch(pl.col("datetime")).dt.convert_time_zone("America/Montevideo")
        )
        lf = lf.cast({"id": pl.Int32, "value": pl.Float32})
        lf.sink_parquet(INTERIM_DATA_DIR / f"{file.stem}.parquet")
