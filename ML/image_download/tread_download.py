# -*- coding: utf-8 -*-
import time
from concurrent.futures import ThreadPoolExecutor

NUM_URL_ACCESS_ATTEMPTS = 3
MAX_WORKERS = 6
WAIT_UNTIL_REPEAT_ACCESS = 2


def thread_download(cutout_service, bands_to_download, fov, ra, dec):
    attempts = 0
    while attempts < NUM_URL_ACCESS_ATTEMPTS:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = {
                    band: executor.submit(
                        cutout_service.download_image, url, fov, ra, dec
                    )
                    for band, url in bands_to_download.items()
                }
            results = {band: r.result() for band, r in results.items()}
            break
        except:
            time.sleep(WAIT_UNTIL_REPEAT_ACCESS)
            results = {}
            attempts += 1
    return results
