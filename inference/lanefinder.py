import cv2
import numpy as np
import time
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from image.processing import preprocessing, postprocessing


class Lanefinder:

    def __init__(self, model_path, input_shape, output_shape, quant, dequant):
        self._window = None
        self._interpreter = self._get_tpu_interpreter(model_path)
        self._cap = cv2.VideoCapture(0)
        self._size = input_shape
        self._output_shape = output_shape
        self._quant = quant
        self._dequant = dequant

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, name):
        self._window = name

    def _get_tpu_interpreter(self, model_path):
        try:
            interpreter = make_interpreter(model_path)
            interpreter.allocate_tensors()
            input_size = common.input_size(interpreter)
            print(f"[INFO] Model input size: {input_size}")
            return interpreter
        except RuntimeError as e:
            print(f"[ERROR] Could not initialize TPU interpreter: {e}")
            return None

    def _preprocess(self, frame):
        # Convert to RGB and resize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(self._size))
        pil_img = Image.fromarray(image)
        return pil_img

    def _postprocess(self, pred_obj, frame, camera_offset_bias):
        return postprocessing(
            pred_obj=pred_obj,
            frame=frame,
            mean=self._quant['mean'],
            std=self._quant['std'],
            in_shape=self._size,
            out_shape=self._output_shape,
            camera_offset_bias= camera_offset_bias

        )

    def stream(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            frmcpy = frame.copy()
            pil_img = self._preprocess(frame)

            if self._interpreter is not None:
                common.set_input(self._interpreter, pil_img)

                self._interpreter.invoke()

                output = self._interpreter.get_tensor(self._interpreter.get_output_details()[0]['index'])
                pred = self._postprocess(output, frmcpy)

            else:
                frmcpy = cv2.resize(frmcpy, self._output_shape)
                pred = cv2.putText(
                    frmcpy,
                    'TPU has not been detected!',
                    org=(self._output_shape[0] // 16, self._output_shape[1] // 2),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=1
                )

            window_name = self._window if self._window else 'default'
            cv2.imshow(window_name, pred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def destroy(self):
        self._cap.release()


class LanefinderFromVideo(Lanefinder):

    def __init__(self, src, model_path, input_shape, output_shape, quant, dequant):
        super().__init__(model_path, input_shape, output_shape, quant, dequant)
        self._cap = cv2.VideoCapture(src)
