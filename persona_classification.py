from keras import Sequential, losses, layers, optimizers, regularizers, utils, models
import numpy as np
import jax.numpy as jnp

batch_size = 32

persona_count: int

model = Sequential([
    # Try to increase the features.
    layers.Input((10, persona_count,)),
    layers.LSTM(1024, activation="tanh", return_sequences=False),
    layers.Dense(1024, activation="tanh"),
    layers.Dense(512, activation="tanh"),
    layers.Dense(256, activation="tanh"),
    layers.Dense(128, activation="tanh"),
    layers.Dense(persona_count, activation="softmax")
])

model.summary()
loss = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
input_kernel = jnp.load("PERSONAS.npy")

length = 1000

for epoch in range(0, 1000):
    message_batch = []
    for i in range(persona_count):
        try:
            # Load this with numpy, jax has memory limits.
            matrix = np.load(f"INDIVIDUAL_MESSAGE_MATRIX_{i}.npy")[epoch * length:(epoch + 1) * length]
        except:
            continue

        for k in range(100):
            if matrix[k * 10: (k + 1) * 10].shape[0] != 10:
                break
            time_batch = []
            for row in matrix[k * 10: (k + 1) * 10].astype(dtype=jnp.float32):
                factor = jnp.linalg.norm(row)
                if jnp.isclose(factor, 0):
                    factor = 1
                multiplied = input_kernel @ row / factor
                time_batch.append(jnp.hstack((multiplied, i)))

            message_batch.append(jnp.asarray(time_batch, dtype=jnp.float32))
        del matrix
        print(f"Persona file {i} completed.")

    np.random.shuffle(message_batch)
    message_batch = jnp.asarray(message_batch)
    print(message_batch.shape)
    print("Shuffle ended.")
    y = message_batch[:, 0, -1]
    print(y.shape)
    print("Labels prepared.")
    x = message_batch[:, :, :-1]
    print("Data prepared.")
    print(x.shape, x[0].shape)
    del message_batch
    model.fit(x, y, batch_size=batch_size, epochs=10)
    del x, y
    model.save(f"./models/matrix_multiplied_model_{epoch}.keras")
