import asyncio
from source.frbstats import FRBPopulation

if __name__ == "__main__":
    data = FRBPopulation()
    print(data.dataframe)