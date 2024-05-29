import os
import argparse
from solver import Solver
from data_loader.mtat_loader import get_audio_loader

def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # audio length
    if config.model_type == 'fcn' or config.model_type == 'crnn':
        config.input_length = 29 * 16000
    elif config.model_type == 'musicnn':
        config.input_length = 3 * 16000
    elif config.model_type in ['sample', 'se', 'short', 'short_res', 'q2l', 'q2l2']:
        config.input_length = 59049
    elif config.model_type == 'hcnn':
        config.input_length = 80000
    elif config.model_type == 'attention':
        config.input_length = 15 * 16000

    # get data loder
    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
									split='TRAIN',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)
    solver = Solver(train_loader, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='q2l',
						choices=['musicnn', 'short_res', 'hcnn', 'q2l'])
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)
