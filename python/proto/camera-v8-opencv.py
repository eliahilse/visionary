import cv2
import numpy as np

# consts
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

# text
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# colors
BLACK = (0,0,0)
BLUE = (255,178,50)
YELLOW = (0,255,255)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    return outputs

def post_process(input_image, output):
    print(f"shape: {output.shape}")
    print(f"type: {output.dtype}")
    print(f"size: {output.size}")

    # output is (1, 84, 8400)
    output = output.reshape((84, -1))
    output = output.T

    boxes = output[:, :4]
    scores = output[:, 4:]

    # Get class id with highest score and the corresponding score
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # confidence filter
    mask = confidences > CONFIDENCE_THRESHOLD
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert boxes to xyxy format and scale to image size
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    boxes[:, [0, 2]] *= input_image.shape[1] / INPUT_WIDTH
    boxes[:, [1, 3]] *= input_image.shape[0] / INPUT_HEIGHT

    # remove non-maximums
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for i in indices:
        box = boxes[i]
        left = int(box[0])
        top = int(box[1])
        width = int(box[2] - box[0])
        height = int(box[3] - box[1])
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        draw_label(input_image, label, left, top)

    return input_image

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# load model file -> create one using build-model
net = cv2.dnn.readNet("yolov8m.onnx")

# init cam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("error getting frame")
        break

    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    cv2.imshow("yolo v8 live", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()