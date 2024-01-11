#! /bin/bash

rm -r runs
cat ../datasets/*smi | sort | uniq | xargs -n 1 python ../compute_emin.py --level xtb --no-relax --skip-store --compute-config ../parsl-configs/local-single-core.py --surge-amount 0
