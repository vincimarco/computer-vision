import polars as pl
import tqdm

from computer_vision.config import INTERIM_DATA_DIR, RAW_DATA_DIR, params


def interim():
    files = list(RAW_DATA_DIR.glob("*.csv.tar.gz"))
    files.sort()
    for file in (pbar := tqdm.tqdm(files, desc="")):
        pbar.set_description(f"Processing {file.name}")
        lf = pl.scan_csv(
            file,
            new_columns=["datetime", "id", "value"],
            ignore_errors=True,
            schema={"datetime": pl.Int32, "id": pl.Int32, "value": pl.Float32},
            low_memory=True,
        )
        lf = lf.cast({"id": pl.Int32, "value": pl.Float32})
        lf = lf.drop_nulls("datetime")
        lf = lf.set_sorted(["datetime", "id"])
        lf = lf.with_columns(
            pl.from_epoch(pl.col("datetime")).dt.convert_time_zone(
                "America/Montevideo"
            ),
        )
        if params["preprocessing"]["to_wh"]:
            lf = lf.with_columns(pl.col("value") * 1000)
        lf.sink_parquet(INTERIM_DATA_DIR / f"{file.stem}.parquet", engine="streaming")
