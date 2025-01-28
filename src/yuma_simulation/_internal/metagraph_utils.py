from yuma_simulation._internal.logger_setup import main_logger as logger

import os
import time
import torch
import bittensor as bt
from multiprocessing import Pool
from .experiment_setup import ExperimentSetup


def ensure_tensor_on_cpu(obj):
    return obj.cpu() if isinstance(obj, torch.Tensor) else obj


def check_file_corruption(file_path):
    """
    True if torch.load or torch.jit.load can read the file, else False.
    """
    try:
        torch.load(file_path, map_location="cpu")
        logger.debug(f"{file_path} loaded with torch.load")
        return True
    except Exception as e:
        logger.warning(f"{file_path} failed torch.load: {e}")
        try:
            torch.jit.load(file_path, map_location="cpu")
            logger.debug(f"{file_path} loaded with torch.jit.load")
            return True
        except Exception as e:
            logger.error(f"{file_path} failed torch.jit.load: {e}")
            return False


def download_metagraph(netuid, start_block, file_prefix, max_retries=5, retry_delay=5):
    """
    1) If file for block already exists and isn't corrupted, we skip.
    2) Otherwise, we download a new one, save, verify; if corrupted, remove and go to next block.
    """
    block = start_block
    for attempt in range(max_retries):
        file_path = f"{file_prefix}_{block}.pt"

        # 1) If file exists and is valid, skip download
        if os.path.isfile(file_path):
            if check_file_corruption(file_path):
                logger.info(f"Block {block}: File already valid, skipping.")
                return block
            else:
                logger.warning(f"Block {block}: Corrupted file found, removing.")
                os.remove(file_path)

        # 2) Download a fresh file
        try:
            logger.info(f"Block {block}: Downloading metagraph --> {file_path}")
            archive_subtensor = bt.subtensor("archive")
            meta = archive_subtensor.metagraph(netuid=netuid, block=block, lite=False)
            meta_dict = {
                "netuid": meta.netuid,
                "block": meta.block,
                "S": ensure_tensor_on_cpu(meta.S),
                "W": ensure_tensor_on_cpu(meta.W),
                "hotkeys": meta.hotkeys,
            }
            torch.save(meta_dict, file_path)

            # 3) Check corruption
            if check_file_corruption(file_path):
                logger.debug(f"Block {block}: File verified successfully.")
                return block
            else:
                logger.error(f"Block {block}: File is corrupted, removing.")
                os.remove(file_path)

        except Exception as e:
            logger.error(f"Block {block}: Error: {e}")

        # Move on to the next block if this one fails
        block += 1
        time.sleep(retry_delay)

    # If we reach here, max_retries are used up
    raise RuntimeError(
        f"Failed to download metagraph for netuid={netuid} "
        f"starting at block={start_block} after {max_retries} retries."
    )


def load_metas_from_directory(storage_path: str) -> list[torch.Tensor]:
    metas = []

    logger.info(f"Checking directory: {storage_path}")
    if not os.path.exists(storage_path):
        logger.error(f"Directory does not exist: {storage_path}")
        return metas

    files = os.listdir(storage_path)
    logger.debug(f"Found files: {files}")

    for filename in sorted(files):
        file_path = os.path.join(storage_path, filename)
        logger.debug(f"Processing file: {file_path}")

        try:
            data = torch.load(file_path, map_location="cpu")
            metas.append(data)
            logger.debug(f"Loaded file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(metas)} metagraphs.")
    return metas


class DownloadMetagraph:
    def __init__(self, setup=None):
        self.setup = setup if setup else ExperimentSetup()

    def run(self):
        os.makedirs(self.setup.metagraph_storage_path, exist_ok=True)
        logger.info(f"Created directory: {self.setup.metagraph_storage_path}")

        # Prepare the (netuid, block, prefix) args
        args = []
        for netuid in self.setup.netuids:
            for conceal_period in self.setup.conceal_periods:
                for dp in range(self.setup.data_points):
                    block_early = self.setup.start_block + self.setup.tempo * dp
                    prefix_early = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_early}",
                    )
                    args.append((netuid, block_early, prefix_early))

                    block_late = self.setup.start_block + self.setup.tempo * (
                        dp + conceal_period
                    )
                    prefix_late = os.path.join(
                        self.setup.metagraph_storage_path,
                        f"netuid{netuid}_block{block_late}",
                    )
                    args.append((netuid, block_late, prefix_late))

        args = list(set(args))  # remove duplicates
        logger.debug(f"Prepared download arguments: {args}")

        with Pool(processes=self.setup.processes) as pool:
            pool.starmap(download_metagraph, args)
        logger.info("DownloadMetagraph run completed.")


if __name__ == "__main__":
    DownloadMetagraph().run()
