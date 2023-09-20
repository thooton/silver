import jax
import jax.numpy as jnp
from flax import linen as nn

def precompute_rope(head_dim, seq_len):
    freqs = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
    freqs = jnp.outer(jnp.arange(seq_len), freqs)
    freqs = jnp.repeat(freqs, 2, axis=-1)
    return jnp.sin(freqs), jnp.cos(freqs)

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    x = x.reshape(*x.shape[:-2], -1)
    return x

def apply_rope(x, seq_len, rope):
    sin, cos = rope
    sin = sin[:seq_len]
    cos = cos[:seq_len]
    return (x * cos) + (rotate_every_two(x) * sin)

def precompute_mask(seq_len):
    mask = jnp.full((seq_len, seq_len), -jnp.inf)
    mask = jnp.triu(mask, 1)
    return mask

class RMSNorm(nn.Module):
    dim: int
    eps: float
    dtype: str
    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, self.dim, self.dtype)
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(
            jnp.square(x).mean(-1, keepdims=True)
            + self.eps
        )
        x = x.astype(self.dtype)
        return x * self.weight

normal_init = jax.nn.initializers.normal(0.02)

class RwkvFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrk = self.param("wrk", normal_init, (self.model_dim, self.model_dim + self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rk = x @ self.wrk
        r = rk[..., :self.model_dim]
        k = rk[..., self.model_dim:]
        k = jnp.square(nn.relu(k))
        v = k @ self.wv
        x = nn.sigmoid(r) * v
        return x

class SwigluFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wab = self.param("wab", normal_init, (self.model_dim, 2 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        a, b = jnp.split(x @ self.wab, 2, axis=-1)
        x = nn.silu(a) * b
        x = x @ self.wo
        return x

class SwiSwigluFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wabc = self.param("wabc", normal_init, (self.model_dim, 3 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        abc = x @ self.wabc
        ab = abc[..., :2*self.ff_dim]
        c = abc[..., 2*self.ff_dim:]
        ab = nn.silu(ab)
        a, b = jnp.split(ab, 2, axis=-1)
        x = a * b * c
        x = x @ self.wo
        return x

class SwiSwiSwigluFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wabcd = self.param("wabcd", normal_init, (self.model_dim, 4 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        abcd = x @ self.wabcd
        abc = abcd[..., :3*self.ff_dim]
        d = abcd[..., 3*self.ff_dim:]
        abc = nn.silu(abc)
        a, b, c = jnp.split(abc, 3, axis=-1)
        x = a * b * c * d
        x = x @ self.wo
        return x

class RegluSquaredFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wab = self.param("wab", normal_init, (self.model_dim, 2 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        a, b = jnp.split(x @ self.wab, 2, axis=-1)
        x = jnp.square(nn.relu(a)) * b
        x = x @ self.wo
        return x

class ReRegluSquaredFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wabc = self.param("wabc", normal_init, (self.model_dim, 3 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        abc = x @ self.wabc
        ab = abc[..., :2*self.ff_dim]
        c = abc[..., 2*self.ff_dim:]
        ab = jnp.square(nn.relu(ab))
        a, b = jnp.split(ab, 2, axis=-1)
        x = a * b * c
        x = x @ self.wo
        return x

class SigRegluSquaredFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrab = self.param("wrab", normal_init, (self.model_dim, self.model_dim + 2 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rab = x @ self.wrab
        r = rab[..., :self.model_dim]
        ab = rab[..., self.model_dim:]
        a, b = jnp.split(ab, 2, axis=-1)
        x = jnp.square(nn.relu(a)) * b
        x = x @ self.wo
        x *= nn.sigmoid(r)
        return x

class ReluCubedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = nn.relu(k)
        k = k * k * k
        v = k @ self.wv
        return v

class RegluCubedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wab = self.param("wab", normal_init, (self.model_dim, 2 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        a, b = jnp.split(x @ self.wab, 2, axis=-1)
        a = nn.relu(a)
        a = a * a * a
        x = a * b
        x = x @ self.wo
        return x

class SigRegluCubedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrab = self.param("wrab", normal_init, (self.model_dim, self.model_dim + 2 * self.ff_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rab = x @ self.wrab
        r = rab[..., :self.model_dim]
        ab = rab[..., self.model_dim:]
        a, b = jnp.split(ab, 2, axis=-1)
        a = nn.relu(a)
        a = a * a * a
        x = a * b
        x = x @ self.wo
        x *= nn.sigmoid(r)
        return x

class ReluFourthedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = nn.relu(k)
        k = jnp.square(k)
        k = jnp.square(k)
        v = k @ self.wv
        return v

class SigReluCubedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrk = self.param("wrk", normal_init, (self.model_dim, self.model_dim + self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rk = x @ self.wrk
        r = rk[..., :self.model_dim]
        k = rk[..., self.model_dim:]
        k = nn.relu(k)
        k = k * k * k
        v = k @ self.wv
        x = nn.sigmoid(r) * v
        return x

class SigReluFourthedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrk = self.param("wrk", normal_init, (self.model_dim, self.model_dim + self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rk = x @ self.wrk
        r = rk[..., :self.model_dim]
        k = rk[..., self.model_dim:]
        k = nn.relu(k)
        k = jnp.square(k)
        k = jnp.square(k)
        v = k @ self.wv
        x = nn.sigmoid(r) * v
        return x

class SigReluEighthedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrk = self.param("wrk", normal_init, (self.model_dim, self.model_dim + self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rk = x @ self.wrk
        r = rk[..., :self.model_dim]
        k = rk[..., self.model_dim:]
        k = nn.relu(k)
        k = jnp.square(k)
        k = jnp.square(k)
        k = jnp.square(k)
        v = k @ self.wv
        x = nn.sigmoid(r) * v
        return x

class CosineFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = k * jnp.sin(5 * k)
        v = k @ self.wv
        return v

class ReluSquaredFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = jnp.square(nn.relu(k))
        v = k @ self.wv
        return v

class ReluFifthedFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = nn.relu(k)
        kk = k
        k = jnp.square(k)
        k = jnp.square(k)
        k *= kk
        v = k @ self.wv
        return v

class SoftmaxFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        k = x @ self.wk
        k = nn.softmax(k)
        v = k @ self.wv
        return v

class AReluSquaredFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wk = self.param("wk", normal_init, (self.model_dim, self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
        self.alpha = self.param("alpha", lambda *_: jnp.array(-0.9, dtype=self.dtype))
        self.beta = self.param("beta", lambda *_: jnp.array(2.0, dtype=self.dtype))
    def __call__(self, x):
        k = x @ self.wk
        relu_k = nn.relu(k)
        a = (relu_k * nn.sigmoid(self.beta)) + nn.relu(-k) * self.alpha
        k = jnp.square(relu_k) + a
        v = k @ self.wv
        return v

class ShortMemFFN(nn.Module):
    model_dim: int
    ff_dim: int
    dtype: str
    def setup(self):
        self.wrk = self.param("wrk", normal_init, (self.model_dim, self.model_dim + self.ff_dim), self.dtype)
        self.wv = self.param("wv", normal_init, (self.ff_dim, self.model_dim), self.dtype)
    def __call__(self, x):
        rk = x @ self.wrk
        r = rk[..., :self.model_dim]
        k = rk[..., self.model_dim:]
        k = jnp.square(nn.relu(k))
        k = k.astype(jnp.float32)
        k1, k2 = jnp.split(k, 2, axis=-1)
        k = jax.lax.complex(k1, k2)
        k /= jax.lax.cummax(k, axis=len(k.shape)-2)
        k = jnp.cumprod(k, axis=-2)
        k = jnp.concatenate([jnp.real(k), jnp.imag(k)], axis=-1)
        k = k.astype(self.dtype)
        v = k @ self.wv
        x = nn.sigmoid(r) * v
        return x

ff_kind_mapping = {
    "rwkv": RwkvFFN,
    "swiglu": SwigluFFN,
    "swiswiglu": SwiSwigluFFN,
    "swiswiswiglu": SwiSwiSwigluFFN,
    "reglu_squared": RegluSquaredFFN,
    "rereglu_squared": ReRegluSquaredFFN,
    "sigreglu_squared": SigRegluSquaredFFN,
    "relu_cubed": ReluCubedFFN,
    "reglu_cubed": RegluCubedFFN,
    "sigreglu_cubed": SigRegluCubedFFN,
    "relu_fourthed": ReluFourthedFFN,
    "sigrelu_cubed": SigReluCubedFFN,
    "sigrelu_fourthed": SigReluFourthedFFN,
    "sigrelu_eighthed": SigReluEighthedFFN,
    "cosine": CosineFFN,
    "relu_squared": ReluSquaredFFN,
    "relu_fifthed": ReluFifthedFFN,
    "softmax": SoftmaxFFN,
    "arelu_squared": AReluSquaredFFN,
    "short_mem": ShortMemFFN
}

class Attention(nn.Module):
    model_dim: int
    query_count: int
    kv_count: int
    dtype: str
    def setup(self):
        assert (self.query_count % self.kv_count) == 0
        self.head_dim = self.model_dim // self.query_count
        self.wqkv = self.param("wqkv", normal_init, (self.model_dim, self.model_dim + 2 * self.kv_count * self.head_dim), self.dtype)
        self.wo = self.param("wo", normal_init, (self.model_dim, self.model_dim), self.dtype)
    def __call__(self, x, mask, rope):
        batch_size, seq_len, model_dim = x.shape
        kv_repeat = self.query_count // self.kv_count
        qkv = x @ self.wqkv
        # (batch_size, seq_len, model_dim)
        q = qkv[..., :self.model_dim]
        # (batch_size, seq_len, kv_repeat, kv_count, head_dim)
        q = q.reshape(batch_size, seq_len, -1, self.kv_count, self.head_dim)
        # (batch_size, kv_repeat, kv_count, seq_len, head_dim)
        q = jnp.moveaxis(q, 1, 3)
        q = apply_rope(q, seq_len, rope)
        # (batch_size, seq_len, kv_count * head_dim)
        k, v = jnp.split(qkv[..., self.model_dim:], 2, axis=-1)
        # (batch_size, seq_len, kv_count, head_dim)
        k, v = (
            t.reshape(batch_size, seq_len, self.kv_count, self.head_dim)
            for t in (k, v)
        )
        # (batch_size, kv_count, seq_len, head_dim)
        k, v = (
            t.swapaxes(1, 2)
            for t in (k, v)
        )
        k = apply_rope(k, seq_len, rope)
        k *= (self.head_dim ** -0.5)
        # (batch_size, 1, kv_count, seq_len, head_dim)
        k, v = (
            jnp.expand_dims(t, 1)
            for t in (k, v)
        )
        # (batch_size, kv_repeat, kv_count, seq_len, seq_len)
        a = q @ k.swapaxes(-2, -1)
        a = a + mask[:seq_len, :seq_len]
        a = nn.softmax(a, axis=-1)
        # (batch_size, kv_repeat, kv_count, seq_len, head_dim)
        x = a @ v
        # (batch_size, query_count, seq_len, head_dim)
        x = x.reshape(batch_size, self.query_count, seq_len, self.head_dim)
        # (batch_size, seq_len, query_count, head_dim)
        x = x.swapaxes(1, 2)
        # (batch_size, seq_len, model_dim)
        x = x.reshape(batch_size, seq_len, model_dim)
        x = x @ self.wo
        return x

class Block(nn.Module):
    model_dim: int
    ff_dim: int
    query_count: int
    kv_count: int
    norm_eps: float
    ff_kind: str
    dtype: str
    def setup(self):
        self.ln_1 = RMSNorm(self.model_dim, self.norm_eps, self.dtype)
        self.attn = Attention(self.model_dim, self.query_count, self.kv_count, self.dtype)
        self.ln_2 = RMSNorm(self.model_dim, self.norm_eps, self.dtype)
        self.ffn = ff_kind_mapping[self.ff_kind](self.model_dim, self.ff_dim, self.dtype)
    def __call__(self, x, mask, rope):
        x += self.attn(self.ln_1(x), mask, rope)
        x += self.ffn(self.ln_2(x))
        return x

time_shift_cat = jax.vmap(
    lambda a, b: jnp.concatenate([a, b], axis=0),
    in_axes=(None, 0),
    out_axes=0
)
class TimeShift(nn.Module):
    dim: int
    amount: int
    dtype: str
    def setup(self):
        self.padding = self.param("padding", normal_init, (self.amount, self.dim), self.dtype)
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        x = x[:, :-self.amount, :]
        x = time_shift_cat(self.padding, x)
        return x

class Silver(nn.Module):
    model_dim: int
    ff_dim: int
    query_count: int
    kv_count: int
    layer_count: int
    norm_eps: float
    seq_len: int
    vocab_size: int
    ff_kind: str
    dtype: str
    def setup(self):
        self.embed = self.param("embed", normal_init, (self.vocab_size, self.model_dim), self.dtype)
        self.blocks = [
            Block(self.model_dim, self.ff_dim, self.query_count, self.kv_count, self.norm_eps, self.ff_kind, self.dtype)
            for _ in range(self.layer_count)
        ]
        self.ln_f = RMSNorm(self.model_dim, self.norm_eps, self.dtype)
        self.mask = precompute_mask(self.seq_len)
        self.rope = precompute_rope(self.model_dim // self.query_count, self.seq_len)
    def __call__(self, x):
        x = jnp.take(self.embed, x, axis=0)
        for block in self.blocks:
            x = block(x, self.mask, self.rope)
        x = self.ln_f(x)
        x = x @ self.embed.T
        return x
