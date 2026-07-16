from experiments.simclr_experiment import SimCLR
import yaml
import argparse
def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="mmwhs")
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pretrained_checkpoint", required=True)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_option()
    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    config['batch_size'] = args.batch_size
    config['epochs'] = args.epoch
    config['config_path'] = args.config
    config['data_dir'] = args.data_dir
    config['pretrained_checkpoint'] = args.pretrained_checkpoint
    print(config)

    simclr = SimCLR(config)
    simclr.train()
