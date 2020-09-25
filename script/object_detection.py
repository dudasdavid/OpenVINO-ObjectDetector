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

        #self.core = None
        self.net = None

        self.core = IECore()

        try:
            # with 2020.1 openvino:
            #self.model = IENetwork(self.model_structure, self.model_weights)
            # with 2020.4 openvino:
            self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

        self.n, self.c, self.h, self.w = self.input_shape

        fh = open("coco-labels-paper.txt","r")
        self.objectNames = fh.readlines()
        fh.close()

        self.numRequests = 4
        self.cur_request_id = 0
        self.next_request_id = int(self.numRequests/2)

        self.firstFrames = True
        self.firstFramesCounter = 0

        

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        

        # Add a CPU extension, if applicable
        if self.cpu_extension and "CPU" in self.device:
            self.core.add_extension(self.cpu_extension, self.device)

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=self.numRequests)


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        # 1) Preprocessing
        height, width, channels = image.shape
        preprocessed_image = self.preprocess_input(image, self.h, self.w)

        # 2) Sync inference
        input_dict = {self.input_name: preprocessed_image}
        # sync API
        #result = self.net.infer(input_dict)[self.output_name]

        # async API
        infer_request = self.net.start_async(self.next_request_id, input_dict)
        infer_status = self.net.requests[self.cur_request_id].wait()
        result = self.net.requests[self.cur_request_id].outputs[self.output_name]

        self.cur_request_id = (self.cur_request_id + 1) % self.numRequests
        self.next_request_id = (self.next_request_id + 1) % self.numRequests


        # 3) post processing inference results
        objects = self.preprocess_output(result, width, height, self.threshold)

        # 4) return coordinates
        return objects

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

    def filter_objects(self, objects, iou_threshold, prob_threshold):
        # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if self.intersection_over_union(objects[i], objects[j]) > iou_threshold:
                    objects[j]['confidence'] = 0

        return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)

    def intersection_over_union(self, box_1, box_2):
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union

    def preprocess_output(self, result, width, height, pred_th):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        objects = []

        if self.firstFrames == False:
            for item in result[0][0]:
                if item[0] == -1:
                    break
                else:
                    #print(item)
                    conf = item[2]
                    if conf >= pred_th:
                        xmin = int(item[3] * width)
                        ymin = int(item[4] * height)
                        xmax = int(item[5] * width)
                        ymax = int(item[6] * height)

                        objects.append({'class': self.objectNames[int(item[1])-1].strip(), 'confidence': item[2], 'xmin': max(0,xmin), 'ymin': max(0,ymin), 'xmax': max(0,xmax), 'ymax': max(0,ymax)})

            objects = self.filter_objects(objects, 0.4, pred_th)

        if self.firstFrames == True:
            self.firstFramesCounter += 1
            if self.firstFramesCounter == self.numRequests:
                self.firstFrames = False

        #print(objects)

        return objects


