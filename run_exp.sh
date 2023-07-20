#!/bin/bash
time=$(date "+%Y-%m-%d_%H-%M-%S")
for seed in 100 200 300 400 500
  do
  for noise in 0.1
    do
    for length in 20
      do
      for trainfreq in 5
        do
        for learning_rate in 5e-2
          do
          python3 main.py -env "PATHTRACK" -se ${seed} -n 10 -N ${length} -tf ${trainfreq} -pstd ${noise} -as ${learning_rate} -ck 10 -cb 128 --eval_freq 10 --topn 1.0 --max_iter 2\
                          -dir "./track/5seed/final_final/${trainfreq}/10_${length}_${trainfreq}_1.0_0.99_0.95_10_128_${noise}_(${learning_rate}-per1e-4)_5e-4_${time}-0/${seed}"
          done
      done
    done
  done
done