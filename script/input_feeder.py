'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2

class InputFeeder:
    def __init__(self, input_type, input_file=None, n_image=1, flip=None):
        '''

        :param input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        :param input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        :param n_image: int, every n-th image is sent to the inference, default is 1, every frame is sent for processing
        :param flip: int, 0 = vertical flip, 1 = horizontal flip, -1 = both
        '''
        self.input_type=input_type
        if input_type=='video' or input_type=='image':
            self.input_file=input_file
        self.n_image = n_image
        self.flip = flip
        self.width = 640
        self.height = 480
        self.fps = 30

    def load_data(self):
        if self.input_type=='video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type=='cam':
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, self.width)
            self.cap.set(4, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        else:
            self.cap=cv2.imread(self.input_file)

    def set_camera_properties(self, width=640, height=480, fps=30):
        '''
        :param width: int, capture width
        :param height: int, capture height
        :param fps: int, capture fps
        '''
        self.width = width
        self.height = height
        self.fps = fps


    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            for _ in range(self.n_image):
                if self.input_type == 'image':
                    # copy is needed to avoid using the same frame buffer
                    frame = self.cap.copy()
                else:
                    _, frame=self.cap.read()

            if self.flip != None:
                frame = cv2.flip(frame, self.flip)

            yield frame



    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()

