from parsl import Config, HighThroughputExecutor

config = Config(executors=[HighThroughputExecutor(label='test')])
