1.跑的代码为：python main.py --detection --config cifar10.yml --doc test

2.需要注意修改：seq_next = list(seq[1:]) + [999]:999为对应seq = range(0, 1000, 100)中间的1000-1