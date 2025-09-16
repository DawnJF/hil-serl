import jax
import jax.numpy as jnp
from flax import linen as nn

# 定义一个简单的线性模型
model = nn.Dense(features=1)

# 生成随机输入和参数
key = jax.random.PRNGKey(0)
dummy_input = jax.random.normal(key, (1, 5))
params = model.init(key, dummy_input)['params']

# 前向传播
output = model.apply({'params': params}, dummy_input)
print("模型输出形状:", output.shape)

# 定义一个简单的损失函数并计算梯度
def loss_fn(params, input):
    output = model.apply({'params': params}, input)
    return jnp.mean(output ** 2)

grads = jax.grad(loss_fn)(params, dummy_input)
print("梯度计算成功。梯度键名:", list(grads.keys()))