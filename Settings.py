import enum
import numpy
import os

class Likelihood(enum.Enum):
    UNKNOWN = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5


class Weight(enum.Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


score_ranges = {'Good': range(int(5.2650 * 1000),int(6.2651 * 1000)), 'Average': range(int(3.665 * 1000), int(5.2650 * 1000)), 'Bad': range(int(3.665 * 1000))}

min_detection_confidence = 0.9
blur_threshold = 35
frames_per_second = 1/20
json_path = os.environ.get('GOOGLE_VISION_API_KEY')
enhancment_median = 127.5
sharpness_factor = 2
ip_cam_url = "http://192.168.0.23:8080/shot.jpg"
weight = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
max_pics_saved = 9
seconds_to_run = 2
number_of_processes = 5
number_of_threads = 5
text_size = 100
text_color = (255, 0, 0)
text_font = "arial.ttf"
save_path = "Generated_Images"
face_ratio = 0.5