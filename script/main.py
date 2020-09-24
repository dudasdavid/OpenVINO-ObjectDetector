from input_feeder import InputFeeder
from object_detection import ObjectDetect
import cv2
import time
from threading import Thread
from queue import Queue
from argparse import ArgumentParser


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required=False, type=str, default="cam",
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so",
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-p", "--precision", type=str, default="FP16",
                        help="Model precision"
                             "(FP16 by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.6,
                        help="Probability threshold for detections filtering"
                        "(0.6 by default)")
    parser.add_argument("-ww", "--width", type=int, default=1920,
                        help="Camera capture width"
                             "(1920 by default)")
    parser.add_argument("-hh", "--height", type=int, default=1080,
                        help="Camera capture height"
                             "(1080 by default)")
    parser.add_argument("-fps", "--fps", type=int, default=30,
                        help="Camera capture fps"
                             "(30 by default)")
    parser.add_argument("-so", "--save_output", type=bool, default=False,
                        help="Save output"
                             "(False by default)")

    parser.add_argument("-mo", "--model", type=str, default="../model/ssd_mobilenet_v2_coco_2018_03_29",
                        help="IR Model"
                             "(ssd_mobilenet_v2_coco_2018_03_29 by default)")    

    return parser

def load_models(args):
    '''
    Loads FP16 or FP32 models
    :param args: argparser arguments
    :return: the loaded model
    '''

    object_model     = args.model

    start_model_load_time = time.time()
    objectDetection = ObjectDetect(object_model, args.device, threshold=args.prob_threshold)
    objectDetection.load_model()
    total_model_load_time = time.time() - start_model_load_time
    print(f"Object detection model load time: {total_model_load_time:.3}s")

    return objectDetection


def infer_on_stream(args, model):
    '''

    :param args: argparser arguments
    :param model: loaded model
    '''

    # get the loaded model instance
    objectDetection = model

    # Handle the input stream
    # Check if the input is a webcam or video or image
    if args.input == 'cam':
        feed = InputFeeder(input_type='cam', flip=1)
        feed.set_camera_properties(args.width, args.height, args.fps)
    elif args.input == 'picam':
        feed = InputFeeder(input_type='picam')
        feed.set_camera_properties(args.width, args.height, args.fps)
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        feed = InputFeeder(input_type='image', input_file=args.input)
    elif args.input.endswith('.mp4'):
        feed = InputFeeder(input_type='video', input_file=args.input)
    else:
        print("ERROR: Invalid input, it must be CAM, image (.jpg, .bmp or .png) or video (.mp4)!")
        raise NotImplementedError

    feed.load_data()

    # run-time switches
    ui_marking = True
    fps_marking = True
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    cv2.namedWindow( "Frame", cv2.WINDOW_NORMAL );
    cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN);

    # Start recording of output saving is enabled
    if args.save_output:
        out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (args.width, args.height))

    for batch in feed.next_batch():
        if batch is None:
            break
        # start measuring overall execution time
        start_processing_time = time.time()
        # 1) First detect objects on the image
        start_object_infer_time = time.time()  # time measurement started
        objects = objectDetection.predict(batch)
        total_object_infer_time = time.time() - start_object_infer_time  # time measurement finished

        # executed only if there are objects on the image
        if len(objects) > 0:

            # if UI marking is turned on draw the vectors, rectangles, etc
            if ui_marking:
                # objects bounding boxes
                for item in objects:
                    # draw the bounding box
                    cv2.rectangle(batch, (item[2], item[3]), (item[4], item[5]), (0, 255, 0), 2)
                    # prepare the label
                    label_text = f"{item[0]}: {item[1]*100:.3}%"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_left = item[2]
                    label_top = item[3] - label_size[1]
                    if (label_top < 1):
                        label_top = 1
                    label_right = label_left + label_size[0]
                    label_bottom = label_top + label_size[1] - 5
                    cv2.rectangle(batch, (label_left - 1, label_top - 6), (label_right + 1, label_bottom + 1),
                              label_background_color, -1)
                    cv2.putText(batch, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


        # Measure overall FPS
        total_processing_time = time.time() - start_processing_time
        total_fps = 1/(total_processing_time)

        # if FPS marking run time switch is turned on print some details on the image
        if fps_marking:
            label_text = f"FPS: {total_fps:.3}"
            cv2.putText(batch, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            label_text = f"Object detection inference time: {total_object_infer_time*1000:.4}ms"
            cv2.putText(batch, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show the output image and save the output video
        cv2.imshow('Frame', batch)
        if args.save_output:
            out.write(batch)

        # Press q on keyboard to exit
        # Press r on keyboard to toggle roll compensation
        # Press u on keyboard to toggle ui drawings
        # Press f on keyboard to fps drawings
        ret = cv2.waitKey(1)
        if ret & 0xFF == ord('q'):
            break
        elif ret & 0xFF == ord('u'):
            ui_marking = not ui_marking
        elif ret & 0xFF == ord('f'):
            fps_marking = not fps_marking

    # close the feed when stopping and finish the video saving
    feed.close()
    if args.save_output:
        out.release()



def main():

    # Grab command line args
    args = build_argparser().parse_args()
    # Load the models
    model = load_models(args)

    # Perform inference on the input stream
    infer_on_stream(args, model)



if __name__ == '__main__':
    main()