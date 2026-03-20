import cv2
import tensorflow as tf
from tensorflow.keras.applications.convnext import preprocess_input
import numpy as np
import json

with open('classes.json') as f:
    class_names = json.load(f)

interpreter = tf.lite.Interpreter(model_path='veg_16.tflite')  # fixed typo
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# preprocesses a video frame and runs it through the TFLite model,
# returning the predicted class name and confidence score.
def predict(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img = preprocess_input(img)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke() # make inference (prediction)
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output[0])
    confidence = output[0][class_index] * 100
    return class_names[str(class_index)], confidence # returns 2 values

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    class_name,confidence = predict(frame)

    cv2.putText(frame, class_name, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                4, (255, 255, 255), 4, 2)

    cv2.imshow('Camera feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()