"""Compute energies of the QM9 set using the energies defined in emin"""
from concurrent.futures import Future
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys
from queue import Queue
from threading import Thread

import pandas as pd
import parsl
from parsl import python_app, Config, HighThroughputExecutor

from emin.parsl import run_molecule, load_config

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--levels', default=['mmff94', 'xtb'], nargs='+',
                        help='Which levels of accuracy to run')
    parser.add_argument('--compute-config',
                        help='Path to the file defining the Parsl configuration. Configuration should be in variable named ``config``')
    args = parser.parse_args()

    # Make the logger
    handlers = [logging.StreamHandler(sys.stdout)]
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    # Start Parsl
    if args.compute_config is not None:
        config = load_config(args.compute_config)
    else:
        config = Config(executors=[HighThroughputExecutor(max_workers=1)])
    parsl.load(config)

    run_app = python_app(run_molecule)
    logger.info('Started Parsl')

    # Load in QM9
    qm9 = pd.read_json('data/qm9.json.gz', lines=True)
    logger.info(f'Loaded {len(qm9)} molecules from QM9')

    # Make the data directory
    data_dir = Path('data/qm9-energies')
    data_dir.mkdir(exist_ok=True, parents=True)

    # Run each level of computation
    for relax in [True, False]:
        for level in args.levels:
            logger.info(f'Starting on {level} relax={relax}')

            # Load what has been done already
            out_path = data_dir / f'{level}-{relax}.csv'
            already_done = set()
            if out_path.exists():
                with out_path.open() as fp:
                    for line in fp:
                        key, _ = line.split(",")
                        already_done.add(key)
            logger.info(f'{len(already_done)} energies have already been computed')

            # Submit what has not
            futures: Queue[tuple[str, Future] | None] = Queue(maxsize=10000)

            def _submit_all():
                for key, smiles in zip(qm9['inchi_key'], qm9['smiles_1']):
                    if key not in already_done:
                        future = run_app(smiles, level, relax)
                        futures.put((key, future))
                futures.put(None)
            submit_thread = Thread(_submit_all)
            submit_thread.start()

            # Write the results as they come in
            success = 0
            with out_path.open('a') as fp:
                while (item := futures.get()) is not None:
                    key, future = item
                    if future.exception() is None:
                        success += 1
                        energy, _ = future.result()
                        print(f'{key},{energy}', file=fp)
            logger.info(f'Completed {success} successfully')
