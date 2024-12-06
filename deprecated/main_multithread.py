import os
import threading

if __name__ = "__main__":
    condition = threading.Condition()

    rl = threading.Thread(name="rl", target=run_rl, args=(condition,))
    deit = threading.Thread(name="deit", target=run_deit, args=(condition,))

    rl.start()
    deit.start()

def run_rl(condition):
    # script running rl
    from rl.actor_critic import main
    main(args, condition)

    
def run_deit():
    # script ruuning deit
    from deit.main import main
    from deit.main import get_args_parser
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, conditon)

