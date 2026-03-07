import pathlib

import polars as pl
import tqdm

from computer_vision.config import FINAL_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR


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
        lf = lf.with_columns(pl.from_epoch(pl.col("datetime")))
        lf = lf.cast({"id": pl.Int32, "value": pl.Float32})
        lf.sink_parquet(INTERIM_DATA_DIR / f"{file.stem}.parquet")


def process_customer_batch(customer_ids):
    lf = pl.scan_parquet(
        INTERIM_DATA_DIR / "*.parquet",
        schema={"datetime": pl.Datetime, "id": pl.Int32, "value": pl.Float32},
    )
    lf = lf.filter(pl.col("id").is_in(customer_ids))
    lf = lf.select(["id", "datetime", "value"])
    lf = lf.sort(["id", "datetime"])
    lf.sink_parquet(
        FINAL_DATA_DIR / f"customers_{customer_ids[0]}_to_{customer_ids[-1]}.parquet",
        engine="streaming",
    )


def final():
    customers_csv = pathlib.Path("data/1.raw/ECDUY/customers.csv")
    pl.scan_csv(customers_csv).sink_parquet(FINAL_DATA_DIR / "customers.parquet")

    customers_ids = (
        pl.scan_csv(customers_csv)
        .select(pl.col("customer_id"))
        .sort("customer_id")
        .collect()
        .to_series()  # ty:ignore[unresolved-attribute]
        .to_list()
    )

    batch_size = 1000
    batches = [
        customers_ids[i : i + batch_size]
        for i in range(0, len(customers_ids), batch_size)
    ]

    for batch in tqdm.tqdm(batches):
        process_customer_batch(batch)

    # multiprocessing.set_start_method("spawn", force=True)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_customer_batch, batch) for batch in batches]
    #     for future in tqdm.tqdm(
    #         concurrent.futures.as_completed(futures), total=len(batches)
    #     ):
    #         future.result()
