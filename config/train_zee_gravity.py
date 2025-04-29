out_dir = 'out-zee-gravity'
eval_interval = 100
eval_iters = 200
log_interval = 1

always_save_checkpoint = True
wandb_log = False
wandb_project = 'physics'
wandb_run_name = 'run1'

dataset = 'zee_gravity'
gradient_accumulation_steps = 5
batch_size = 12
block_size = 256  # smaller blocks are fine for physics text

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-5

beta2 = 0.99
warmup_iters = 100

device = 'cuda'  # or 'cpu'