
from . import config
from . import model
from . import dataLoader
import tensorflow as tf

if __name__ == '__main__':

    training = dataLoader.get_training_dataset()
    evaluation = dataLoader.get_eval_dataset()

    model = model.UNet(input_shape=(config.Kernel_shape[0], config.Kernel_shape[1], 6), nclasses=2)

    model.compile(
		optimizer=tf.keras.optimizers.get(config.Optimizer),
		loss=tf.keras.losses.get(config.Loss),
		metrics=[tf.keras.metrics.get(metric) for metric in config.Metrics])

    model.fit(
        x=training,
        epochs=config.Epochs, 
        steps_per_epoch=int(1000*6/16),
        validation_data=evaluation,
        validation_steps=300*6,
        callbacks=[tf.keras.callbacks.TensorBoard(config.Logs_Dir)]
        )

    model.save(config.Model_Dir, save_format='tf')