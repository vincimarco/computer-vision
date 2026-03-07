import cyclopts

app = cyclopts.App()

preprocessing_app = app.command(cyclopts.App(name="preprocessing"))


@preprocessing_app.command()
def interim():
    from .preprocessing import interim

    interim()


@preprocessing_app.command()
def final():
    from .preprocessing import final

    final()
