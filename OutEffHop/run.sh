module load cuda/12.6.2-gcc-12.4.0
export HF_HOME="/projects/p32013/.cache/" 


source ~/.bashrc && conda activate outlier && which python && python -V



export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export CUDA_HOME=/software/cuda/cuda-12.1.0 

~/.conda/envs/outlier/bin/python -m accelerate.commands.launch --config_file accelerate_configs/2gpu_fp16.yaml run_clm_ddp.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--run_name test_vanilla_llama3_1b \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir /scratch/hlv8980/residual/.hf_data \
--model_cache_dir /scratch/hlv8980/residual/.hf_cache \
--model_type llama \
--tokenizer_name meta-llama/Llama-3.2-1B \
--max_seq_length 2048 \
--block_size 1024 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 100000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 5000 \
--tb_scalar_log_interval 10000 \
--tb_hist_log_interval 20000 \
--model_name_or_path meta-llama/Llama-3.2-1B \
--attn_softmax vanilla \
--attn_res_softmax_fn vanilla \
--output_dir /scratch/hlv8980/residual/output/vanilla_llama3 
