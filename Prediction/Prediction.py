import tensorflow as tf
from Model.Model import create_model
from Utils.Anchor import AnchorBox
from Utils.Coordinates import convert_to_corners
from Data.Preprocessing import resize_and_pad_image
from Data.Data import get_label_info
from glob import glob
import cv2
from Utils.Drawer import visualize_detections
import os

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


def get_inference_model(num_classes):
    """
    ## Loading weights
    """
    # Change this to `model_dir` when not using the downloaded weights
    weights_dir = "data"
    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    """
    ## create model
    """
    model = create_model(num_classes)
    model.load_weights(latest_checkpoint)
    """
    ## Building inference model
    """
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model


def predict_sequences(seq_path, inference_model):

    """
    ## Generating detections
    """

    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    int2str = get_label_info()
    img_path_list = glob(f"{seq_path}\*.png")

    for img_path in img_path_list:
        image = cv2.imread(img_path)
        image_name = os.path.basename(img_path)
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = tf.cast(input_image, dtype=tf.float32)
        input_image, ratio = prepare_image(input_image)
        detections = inference_model.predict(input_image)
        num_detections = detections.valid_detections[0]
        class_names = [
            int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        ]
        image_draw = visualize_detections(
            image,
            detections.nmsed_boxes[0][:num_detections] / ratio,
            class_names,
            detections.nmsed_scores[0][:num_detections],
        )

