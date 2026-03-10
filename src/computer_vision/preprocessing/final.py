import polars as pl
import tqdm

from computer_vision.config import FINAL_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR


def process_customer_batch(customer_ids):
    lf = pl.scan_parquet(
        INTERIM_DATA_DIR / "*.parquet",
        schema={
            "datetime": pl.Datetime("us", "America/Montevideo"),
            "id": pl.Int32,
            "value": pl.Float32,
        },
    )
    lf = lf.set_sorted("datetime", "id")
    lf = lf.filter(pl.col("id").is_in(customer_ids))
    lf = lf.select(["id", "datetime", "value"])
    lf = lf.sort(["id", "datetime"])
    lf.sink_parquet(
        FINAL_DATA_DIR / f"customers_{customer_ids[0]}_to_{customer_ids[-1]}.parquet",
        engine="streaming",
    )


def final():
    customers_csv = RAW_DATA_DIR / "customers.csv"
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

    for batch in (pbar := tqdm.tqdm(batches)):
        pbar.set_description(f"Processing customers {batch[0]} to {batch[-1]}")
        process_customer_batch(batch)

    # multiprocessing.set_start_method("spawn", force=True)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_customer_batch, batch) for batch in batches]
    #     for future in tqdm.tqdm(
    #         concurrent.futures.as_completed(futures), total=len(batches)
    #     ):
    #         future.result()
