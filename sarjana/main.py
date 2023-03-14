# TUI
import rich
import typer

from sarjana import commands

app = typer.Typer()


@app.command()
def debug(frb: str = typer.Argument(...)):
    """Misc functions to be debug"""

# if __name__ == "__main__":
#     from sarjana.handlers import ParquetWaterfall
#     from sarjana.signal import find_burst
#     frb = r"D:\home\datasets\sarjana\raw\wfall\FRB20180725A_waterfall.h5.parquet"
#     rich.print(frb)
#     data = ParquetWaterfall(frb)
#     find_burst(data.ts)


@app.command()
def plot(
    profile: str = typer.Argument(
        ..., help="Path to flux profile data (in `.parquet`)"
    ),
    embedding: str = typer.Argument(..., help="Path to embedding data (in `.csv`)"),
    savefile: str = typer.Argument(
        ..., help="The name of the saved plot file. No extension in name."
    ),
    size: int = typer.Option(30, help="The number of FRB in each plot."),
    peaks: bool = typer.Option(False, help='Whether to show peaks')
):
    """Plots a FacetGrid of flux profiles of each FRB based on categories defined in embedding file."""
    commands.plot_many_flux_profile_by_clustering_groups(
        profile, embedding, savefile, size, find_peaks=peaks
    )


@app.command()
def download(
    eventnames: typer.FileText = typer.Argument(
        ..., help="A newline delimited `.txt` file listing eventnames."
    ),
    tofile: str = typer.Option(
        None, help="Filename in `.parquet` to collect downloaded data into."
    ),
    path: str = typer.Argument(".", help="Download file to this path"),
    limit: int = typer.Option(None, help="How many to download"),
):
    """Download waterfall data from CHIME/FRB database"""
    commands.download_waterfall_data_from_chimefrb_database(
        eventnames=eventnames, tofile=tofile, path=path, limit=limit
    )


@app.command()
def combine(
    eventnames: typer.FileText = typer.Argument(
        ..., help="A newline delimited `.txt` file listing eventnames."
    ),
    collectionfile: str = typer.Argument(
        ..., help="Filename in `.parquet` to collect downloaded data into."
    ),
    filepattern: str = typer.Option(
        "{}_waterfall.h5.parquet", help="File pattern to search for parquet."
    ),
):
    """Combine files into a single parquet file"""
    commands.combine_multifile_into_single_parquet_file(
        eventnames=eventnames, collectionfile=collectionfile, filepattern=filepattern
    )
