import argparse
import os

from train import train
from test import test

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, default=os.getcwd())
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--calibration", type=bool, default=False)
    parser.add_argument("--depth_config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--pepoch", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        assert args.train != args.test
    except AssertionError:
        raise AssertionError("One of the Train and Test Flags must be True. \
Both cannot be True or False")
    
    if args.train == True:
        train(args)
    elif args.test  == True:
        test(args)
    else: pass
        
    print("[Program Termination]")