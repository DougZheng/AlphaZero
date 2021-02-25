import sys
sys.path.append('..')
sys.path.append('../src')

from train import TrainPipeline
import config

if __name__ == '__main__':
    pipeline = TrainPipeline(config.config)
    pipeline.learn()