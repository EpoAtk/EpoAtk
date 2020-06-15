greedy_edges=128
drop_rate=0
date='20200611'
re_rate=0.7
init_alive_numbers=1
tar_hidden=16
fake_ratio=0.03
loss='marginminloss'
modified_graph_filename=$date'_'$init_alive_numbers'populations_'$re_rate'edge_mutation_'$greedy_edges'edge_'$drop_rate'drop_sl_cora_'$tar_hidden'_'$loss'.npy'
save_file='./results/'$date'_'$init_alive_numbers'populations_'$re_rate'edge_inverserank_mut_cro_'$greedy_edges'edge_'$drop_rate'drop_sl_cora_'$tar_hidden'_'$loss'.log'
CUDA_VISIBLE_DEVICES=3 python attack_B.py \
 --greedy_edges $greedy_edges \
 --drop_rate $drop_rate \
 --tar_hidden $tar_hidden \
 --fake_ratio $fake_ratio \
 --re_rate $re_rate \
 --init_alive_numbers $init_alive_numbers \
 --modified_graph_filename $modified_graph_filename \
> $save_file
