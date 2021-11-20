import numpy as np
import tensorflow as tf
import cv2
import time
from PIL import Image


def raw_image(path_image):
    img = cv2.imread(path_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input = cv2.resize(img, (416, 416))
    input = input / 255.

    images_data = []
    for i in range(1):
        images_data.append(input)
    return np.asarray(images_data).astype(np.float32), img


def get_name_classes(path_names):
    with open(path_names, 'r') as f:
        classes = f.read().splitlines()
    return classes


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape=tf.constant([416, 416])):
    scores_max = tf.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = tf.boolean_mask(box_xywh, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

    box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)

    input_shape = tf.cast(input_shape, dtype=tf.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def detect(path_model, img):

    interpreter = tf.lite.Interpreter(model_path=path_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)

    start_time = time.time()
    interpreter.invoke()
    print("--- %s seconds ---" % (time.time() - start_time))

    prediction = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    boxes, conf = filter_boxes(prediction[0], prediction[1], score_threshold=0.25,
                               input_shape=tf.constant([416, 416]))

    return tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            conf, (tf.shape(conf)[0], -1, tf.shape(conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )


def post_process(img, data, names):
    num_classes = len(names)
    h, w, _ = img.shape
    boxes, scores, classes, num_boxes = data

    for i in range(num_boxes[0]):
        if int(classes[0][i]) < 0 or int(classes[0][i]) > num_classes: continue
        coords = boxes[0][i]
        a = int(coords[0] * h)
        c = int(coords[2] * h)
        b = int(coords[1] * w)
        d = int(coords[3] * w)

        score = scores[0][i]
        index = int(classes[0][i])
        box_thick = int(0.6 * (h + w) / 600)
        text = '%s: %.2f' % (names[index], score)

        c1, c2 = (b, a), (d, c)

        cv2.rectangle(img, c1, c2, color=(0, 255, 0), thickness=box_thick)

        cv2.putText(img, text, (b, a - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color=(0, 255, 0), thickness=box_thick//2)
    return img


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = "./checkpoints/yolov4_v1_darknet-416_fp16.tflite"
    names = get_name_classes("./yolo-files/coco.names")
    preprocessed, original = raw_image("test.jpg")
    out = detect(model, preprocessed)
    result = post_process(original, out, names)
    image = Image.fromarray(result.astype(np.uint8))
    image.show()

    # for i in range(1):
    #     print(i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
