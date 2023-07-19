import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_p2p(rank, size):
  """Distributed func: blocking point-to-point communication"""
  tensor = torch.zeros(1)
  if rank == 0:
    tensor += 1
    # send the tensor to process 1
    dist.send(tensor=tensor, dst=1)
  else:
    # receive tensor from process 0
    dist.recv(tensor=tensor, src=0)
  print('Rank ', rank, 'data: ', tensor[0])


def run_collectives(rank, size):
  """Distributed func: collective communication of All-reduce"""
  group = dist.new_group([0, 1])
  tensor = torch.ones(1)
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
  print("Rank ", rank, "data: ", tensor[0])


def init_process(rank, size, fn, backend="gloo"):
  """Initialize distributed environment."""
  # every process will communicate through this master
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '26500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, size)



if __name__ == "__main__":
  size = 2
  processes = []
  mp.set_start_method("spawn")
  for rank in range(size):
    # spawn a process that each sets up the distributed env,
    # init process group, and execute run function.
    p = mp.Process(target=init_process, args=(rank, size, run_collectives))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()

