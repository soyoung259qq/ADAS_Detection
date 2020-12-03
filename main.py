import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from Model.Model import create_model
from Data.Data import get_train_data
from Train.Train import run_train
from Prediction.Prediction import get_inference_model, predict_sequences


def train():

    """
    ## get train, valid data
    """
    num_classes = 80
    batch_size = 1
    data_train, data_valid = get_train_data(batch_size)

    """
    ## create model
    """
    model = create_model(num_classes)

    """
    ## Training the model
    """
    epochs = 10
    run_train(model, num_classes, data_train, data_valid, epochs, 400)


def test():
    num_classes = 80
    seq_path = r"D:\Dataset\Kitti_Object\\tracking\data_tracking_image_2\training\image_02\0019"
    inference_model = get_inference_model(num_classes)
    inference_model.summary()
    predict_sequences(seq_path, inference_model)


if __name__ == '__main__':

    mode = "train"

    if mode is "train":
        train()
    elif mode is "test":
        test()