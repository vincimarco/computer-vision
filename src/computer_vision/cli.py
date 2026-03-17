from typing import Literal

import cyclopts

app = cyclopts.App()

preprocessing_app = app.command(cyclopts.App(name="preprocessing"))


@preprocessing_app.command()
def interim(dataset: Literal["ECD-UY", "UK-DALE"]):
    if dataset == "ECD-UY":
        from .preprocessing.interim import interim

        interim()
    elif dataset == "UK-DALE":
        from .preprocessing.interim import interim_uk_dale

        interim_uk_dale()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


@preprocessing_app.command()
def final(dataset: Literal["ECD-UY", "UK-DALE"]):
    from .preprocessing.final import final

    final()


@app.command()
def train():
    from .train import train

    train()
