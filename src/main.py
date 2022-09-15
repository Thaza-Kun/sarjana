import asyncio
import fetch

if __name__ == "__main__":
    data = asyncio.run(fetch.frbstats.get_frb_population_data_as_csv())
    print(data)