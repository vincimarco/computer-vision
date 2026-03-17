import polars as pl
import tqdm

from computer_vision.config import INTERIM_DATA_DIR, RAW_DATA_DIR


def interim():
    raw_dir = RAW_DATA_DIR / "ECD-UY"
    interim_dir = INTERIM_DATA_DIR / "ECD-UY"
    interim_dir.mkdir(parents=True, exist_ok=True)

    files = list(raw_dir.glob("*.csv.tar.gz"))
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
            pl.col("value") * 1000,
        )
        lf.sink_parquet(INTERIM_DATA_DIR / f"{file.stem}.parquet", engine="streaming")


def interim_uk_dale():
    raw_dir = RAW_DATA_DIR / "UKDALE"
    interim_dir = INTERIM_DATA_DIR / "UKDALE"
    interim_dir.mkdir(parents=True, exist_ok=True)

    files = [
        raw_dir / "house_1" / "channel_1.dat",
        raw_dir / "house_2" / "channel_1.dat",
        raw_dir / "house_3" / "channel_1.dat",
        raw_dir / "house_4" / "channel_1.dat",
        raw_dir / "house_5" / "channel_1.dat",
    ]
    for file in (pbar := tqdm.tqdm(files, desc="")):
        house_id = file.parent.name.split("_")[1]
        pbar.set_description(f"Processing {file.name}")
        lf = pl.scan_csv(
            file,
            has_header=False,
            separator=" ",
            new_columns=["timestamp", "value"],
        )
        lf = lf.with_columns(pl.from_epoch(pl.col("timestamp")))
        lf.sink_parquet(interim_dir / f"{house_id}.parquet", engine="streaming")
