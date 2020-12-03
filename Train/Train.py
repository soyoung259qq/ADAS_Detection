import tensorflow as tf
import os
from Utils.Label import LabelEncoder
from Utils.Losses import RetinaNetLoss



def run_train(model, num_classes, data_train, data_valid, epochs, train_sample_num=0):

    model_dir = "retinanet/"
    """
    ## Set learning rate
    """
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    """
    ## Set optimizer and retina loss function
    """
    loss_fn = RetinaNetLoss(num_classes)
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    """
    ## Setting up callbacks
    """

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    if train_sample_num <= 0:
        model.fit(
            data_train,
            validation_data=data_valid.take(50),
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
        )
    else:
        model.fit(
            data_train.take(train_sample_num),
            validation_data=data_valid.take(50),
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
        )