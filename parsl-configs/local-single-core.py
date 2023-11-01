"""Single worker for each core on your local computer"""
from parsl import Config, HighThroughputExecutor

config = Config(executors=[HighThroughputExecutor(label='test', cores_per_worker=1, cpu_affinity='block')])
