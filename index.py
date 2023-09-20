import wandb
import jax
from jax import numpy as jnp
import numpy as np
import secrets
import optax
import flax
from flax.training.train_state import TrainState
import silver
from flax.training import checkpoints
flax.config.update("flax_use_orbax_checkpointing", False)

conf = type("", (), dict(
    model_dim=768,
    ff_dim=2688,
    query_count=12,
    kv_count=4,
    layer_count=10,
    norm_eps=1e-6,
    seq_len=1024,
    vocab_size=51200,
    ff_kind="short_mem",
    dtype="bfloat16",
    
    device_batch_size=16,
    device_count=4,
    replica_count=4,
    train_file="./train_1.bin",
    
    lr=3e-4 / 3,
    weight_decay=0.1 * 3,
    grad_clip=1.0
))

batch_len = conf.device_count * conf.device_batch_size * conf.seq_len
print(f"batch len: {batch_len:_} tokens/step")

model = silver.Silver(
    conf.model_dim,
    conf.ff_dim,
    conf.query_count,
    conf.kv_count,
    conf.layer_count,
    conf.norm_eps,
    conf.seq_len,
    conf.vocab_size,
    conf.ff_kind,
    conf.dtype
)

params = model.init(jax.random.PRNGKey(secrets.randbits(32)), jnp.ones((1, 8), dtype="uint8"))["params"]
print(f"param count: {sum(t.size for t in jax.tree_util.tree_leaves(params)):_}")

train_data = np.memmap(conf.train_file, dtype=np.uint16, mode="r")
end_index = len(train_data) - batch_len
def get_batch():
    start_index = secrets.randbits(64) % end_index
    batch = train_data[start_index:start_index+batch_len]
    batch = batch.reshape(conf.device_count, conf.device_batch_size, conf.seq_len)
    return jnp.array(batch)

if conf.lr_scheduling:
    schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=conf.lr,
      warmup_steps=conf.lr_warmup_steps,
      decay_steps=conf.lr_decay_steps,
      end_value=conf.lr / 10,
    )

optimizer = optax.chain(
    optax.clip(conf.grad_clip),
    optax.lion(
        learning_rate=conf.lr,
        weight_decay=conf.weight_decay
    )
)

state = flax.training.train_state.TrainState.create(
    apply_fn=model.apply,
    tx=optimizer,
    params=params
)

del params

def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[..., :-1, :],
            batch[..., 1:]
        ).mean()
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)
    return new_state, loss
parallel_train_step = jax.pmap(train_step, "batch")

state = flax.jax_utils.replicate(state)

process_index = jax.process_index()
if process_index == 0:
    wandb.init(project="silver-102m", name="short_mem")

step = 0
while True:
    batch = get_batch()
    state, loss = parallel_train_step(state, batch)
    loss = loss.mean()
    print(f"Step {step}, Loss {loss}")
    if process_index == 0:
      wandb.log({"loss": loss, "tokens": step * batch_len * conf.replica_count})
    step += 1
