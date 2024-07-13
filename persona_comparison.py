import numpy as np
import jax.numpy as jnp

scorer = lambda x: 1 / (x + jnp.exp(-x))

embeds = []
PERSONA_COUNT: int
EMBEDDING_FILE: str
TARGET_FILE: str

for k in range(PERSONA_COUNT):
    matrix = np.load(EMBEDDING_FILE)[:30000]
    norms = jnp.linalg.norm(matrix, axis=1)
    embeds.append(matrix.T / norms)

epsilon = 1e-07  # So that no infinity will be generated in the logarithm
scores = []
for k in range(PERSONA_COUNT):
    temp = []
    for l in range(PERSONA_COUNT):
        diag = -jnp.sum(jnp.log(jnp.sum(jnp.multiply(embeds[k], embeds[l]), axis=0) + epsilon), axis=0)
        temp.append(scorer(jnp.sum(diag)))
    scores.append(temp)

scores = np.asarray(scores)
print(scores)

np.save(TARGET_FILE, scores)
