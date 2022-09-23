import requests
import logging

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
