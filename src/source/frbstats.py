import asyncio
import logging
from pathlib import Path

import pandas as pd

from .external import request_data

from config.data import DATAPATH

class FRBPopulation:
    def __init__(self, output = 'population.csv') -> None:
        asyncio.run(self.download_csv(url='https://www.herta-experiment.org/frbstats/catalogue.csv', output_file=output))

    async def download_csv(self: object, url:str='https://www.herta-experiment.org/frbstats/catalogue.csv', output_file:str='population.csv') -> str:
        result = await request_data(url)
        output_file = Path.joinpath(DATAPATH, output_file)
        with open(output_file, 'w') as f:
            f.write(result.content.decode(result.encoding))
            logging.debug(f'Writing data to {output_file}')
        self.dataframe: pd.Dataframe = pd.read_csv(output_file)
        return output_file