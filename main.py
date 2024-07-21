import tensorflow as tf
from tensorflow import keras
from data_pipeline import analyze_dataset, get_data
from diffusion_model import DiffusionModel
import os

# train for >20 epochs for ok results, >50 for better results
num_epochs = 20

# load datasets
train_dataset, val_dataset = get_data()
image_size = train_dataset._flat_shapes[0][0]

# TODO: Have a look at the dataset. You can (and should) comment this out afterwards.
analyze_dataset(train_dataset, val_dataset)

# training
# create and compile the model
model = DiffusionModel(image_size, widths=[32, 64, 96, 128], block_depth=2)

optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
mae_loss = tf.keras.losses.MeanAbsoluteError()
model.compile(optimizer=optimizer,
              loss=mae_loss)

# save the best model based on the validation KID metric
checkpoint_dir = './checkpoints'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_filepath = os.path.join(checkpoint_dir, 'model_weights_best.weights.h5')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                         monitor="val_kid", mode="min", save_best_only=True)

# calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_dataset)

# run training and plot generated images periodically
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset,
          callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images), checkpoint_callback])

# load the best model and generate images
model.load_weights(checkpoint_filepath)
model.plot_images(epoch=num_epochs+1)
