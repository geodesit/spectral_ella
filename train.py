import pandas as pd
import tensorflow as tf
from important_func import *


BATCH_SIZE = 16
EPOCHS = 100

sig_dict = get_sig_dict("D:\\spectral_proj\\spectral\\sig_online")
CLASSES_NUM = len(sig_dict.keys())
BANDS_NUM = len(max(sig_dict.values(), key=len))
sig_dict = change_len(sig_dict)
x_train = list(itertools.chain(*sig_dict.values()))
y_train = get_tags(sig_dict)
n_samples = np.array(x_train).shape[0]


n_steps_per_epoch = n_samples // BATCH_SIZE
n_steps = n_steps_per_epoch * EPOCHS

indices = tf.range(start=0, limit=tf.shape(x_train)[0], dtype=tf.int32)
idx = tf.random.shuffle(indices)
x_train = tf.gather(x_train, idx)
y_train = tf.gather(y_train, idx)

inputs = tf.keras.layers.Input(BANDS_NUM)
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
# 1 because of its a precent between 0 to 1
# for multi choices enter number of classes and change activation to relu from sigmoid
outputs = tf.keras.layers.Dense(CLASSES_NUM, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
# check initial_learning_rate value
# check weight_decay value
# TODO: Need to find error for function
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.01, decay_steps=n_steps)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule, weight_decay=0.0001)
# loss='sparse_categorical_crossentropy' for a few class
if CLASSES_NUM > 2:
    loss_mode = 'binary_crossentropy'
else:
    loss_mode = 'sparse_categorical_crossentropy'
model.compile(loss=loss_mode, optimizer=optimizer, metrics=['accuracy'])

checkpoint_path = "training/cp-{epoch:04d}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, verbose=1)

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    callbacks=[cp_callback])
print(history.history)

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model.save('white.h5')
