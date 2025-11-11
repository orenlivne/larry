import tensorflow as tf, numpy as np

model = tf.keras.models.load_model("data/maia-2200.h5")
print("✅ Loaded Maia-2200 model.")

# Load your dataset
x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

model.fit(x_train, y_train, epochs=3, batch_size=256)
model.save("models/larry_maia.h5")

print("🎯 Larry fine-tuned model saved to models/larry_maia.h5")
