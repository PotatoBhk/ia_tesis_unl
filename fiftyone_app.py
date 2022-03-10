import fiftyone
import fiftyone.utils.yolo

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="test",
    label_types=["detections"],
    classes=["person"],
    max_samples=538,
)

dataset.name = "coco-2017-test-example"
dataset.persistent = True

# Visualize the dataset in the FiftyOne App
session = fiftyone.launch_app(dataset)