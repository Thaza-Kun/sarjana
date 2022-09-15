from pathlib import Path
import requests
import logging

from config.data import DATAPATH

async def request_data(url, **kwargs) -> requests.Response:
    """Get data from an API url

    Args:
        url (str): The URL to get data, preferably an API endpoint

    Raises:
        requests.RequestException: Raise exception if status code is not 200

    Returns:
        requests.Response: Response object
    """
    logging.debug(f"Awaiting data from {url}")
    result = requests.get(url, **kwargs)
    if result.status_code != 200:
        raise requests.RequestException(result.status_code)
    return result

async def get_frb_population_data_as_csv(url:str='https://www.herta-experiment.org/frbstats/catalogue.csv', output_file='data.csv') -> str:
    result = await request_data(url)
    output_file = Path.joinpath(DATAPATH, output_file)
    with open(output_file, 'w') as f:
        f.write(result.content.decode(result.encoding))
        logging.debug(f'Writing data to {output_file}')
    return output_file