from openvino.inference_engine import IENetwork, IECore
import cv2

class ObjectDetect:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold=0.60):
        '''
        initialize class variables
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.cpu_extension = extensions

        self.core = None
        self.net = None

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

        self.n, self.c, self.h, self.w = self.input_shape

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()

        # Add a CPU extension, if applicable
        if self.cpu_extension and "CPU" in self.device:
            self.core.add_extension(self.cpu_extension, self.device)

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        # 1) Preprocessing
        height, width, channels = image.shape
        preprocessed_image = self.preprocess_input(image, self.h, self.w)

        # 2) Sync inference
        input_dict = {self.input_name: preprocessed_image}
        result = self.net.infer(input_dict)[self.output_name]

        # 3) post processing inference results
        coords = self.preprocess_output(result, width, height, self.threshold)

        # 4) return coordinates
        return coords

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, input_image, height, width):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image = cv2.resize(input_image, (width, height))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)

        return image

    def preprocess_output(self, result, width, height, pred_th):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []

        for box in result[0][0]:  # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= pred_th:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                # max is needed to avoid negative box coordinate that can cause a crash during roi image resize
                coords = [max(0,xmin), max(0,ymin), max(0,xmax), max(0,ymax)]

        # print(coords)
        # print("----")
        # print(coords[0])
        return coords
