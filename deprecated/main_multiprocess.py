import time
from multiprocessing import Process, Queue

def run_rl(lst_queue):
    from rl.sac.sac import run
    run(lst_queue)
    pass

def run_deit(lst_queue):
    pass

def main():
    lst_queue = []
    queue_mask = Queue()
    queue_state = Queue()
    queue_reset = Queue()
    lst_queue.append(queue_mask)
    lst_queue.append(queue_state)
    lst_queue.append(queue_reset)

    p_rl = Process(target=run_rl, args=(lst_queue,))
    p_deit = Process(target=run_deit, args=(lst_queue))

    p_rl.start()
    p_deit.start()

    p_rl.join()
    p_deit.join()

    p_rl.terminate()
    p_deit.terminate()

if __name__ == '__main__':
    main()
