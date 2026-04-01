import cyclopts

app = cyclopts.App()

preprocessing_app = app.command(cyclopts.App(name="preprocessing"))


@preprocessing_app.command()
def interim():
    from .preprocessing.interim import interim

    interim()


@preprocessing_app.command()
def final():
    from .preprocessing.final import final

    final()


@app.command()
def train():
    from .train import train

    train()


@app.command()
def test():
    from .test import test

    test()


@app.command()
def evaluate():
    from .evaluate import eval

    eval()


@app.command()
def plot():
    from .plot import plot

    plot()


@app.command()
def export():
    from .export import export

    export()


@app.command()
def forecast():
    from .forecast import forecast

    forecast()
