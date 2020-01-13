from google.cloud import vision
from google.cloud.vision import types
import urllib.request
import threading
from queue import Queue
import time
from collections import OrderedDict
import cv2
import numpy as np
from PIL import ImageDraw , ImageStat, Image, ImageEnhance, ImageFont
import io
import shutil
import os
import wx
import concurrent.futures as cf
import Settings


def get_faces_data(faces):
    faces_data = list()
    for face in faces:
        width = face.bounding_poly.vertices[1].x - face.bounding_poly.vertices[0].x 
        height = face.bounding_poly.vertices[-1].y - face.bounding_poly.vertices[0].y
        faces_data.append([face, (width, height)])

    return faces_data


def get_valid_faces(faces_data):
    faces_data = sorted(faces_data, key=lambda k: k[1], reverse=True)
    valid_faces = [faces_data[0][0]]

    main_face = faces_data[0][1]
    for face_data in faces_data[1:]:
        if face_data[1][0] / main_face[0] >= Settings.face_ratio:
            valid_faces.append(face_data[0])
    return valid_faces


def write_to_image(text, image, y_cordinate):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(Settings.text_font, Settings.text_size)
    draw.text((0,y_cordinate),str(text),Settings.text_color,font=font)


def variance_of_laplacian(image):
    try:
        variance = cv2.Laplacian(image, cv2.CV_64F).var()
        return variance
    except Exception:
        print ('Error finding Laplacian variance')
 
def detect_faces(image_content):
    global BLUR_THRESHOLD
    image = types.Image(content=image_content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    
    nparr = np.frombuffer(image_content, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if variance_of_laplacian(img_np) < Settings.blur_threshold: faces = []

    return image_content, faces
 
 
def check_smile(faces, labels=None):
    pic_valid = False
    for face in faces:
        if face.joy_likelihood >= Settings.Likelihood.POSSIBLE.value and \
           face.anger_likelihood <= Settings.Likelihood.UNLIKELY.value and \
           face.sorrow_likelihood <= Settings.Likelihood.UNLIKELY.value and \
           face.under_exposed_likelihood <= Settings.Likelihood.UNLIKELY.value and \
           face.blurred_likelihood <= Settings.Likelihood.UNLIKELY.value and \
           face.detection_confidence >= Settings.min_detection_confidence and \
           all([abs(angle) <= 15 for angle in [face.roll_angle, face.pan_angle, face.tilt_angle]]):
           pic_valid = True
        else:
            pic_valid = False
            break
 
    return pic_valid
 
 
def face_score(face):
    score = (face.joy_likelihood / 5) * Settings.Weight.FIVE.value
    score -= (face.sorrow_likelihood / 5) * Settings.Weight.FIVE.value
    score -= (face.anger_likelihood / 5) * Settings.Weight.ONE.value
    score -= (face.under_exposed_likelihood / 5) * Settings.Weight.ONE.value 
    score -= (face.blurred_likelihood / 5) * Settings.Weight.ONE.value
    score += (face.detection_confidence) * Settings.Weight.THREE.value
    score -= (abs(face.roll_angle) / 90) * Settings.Weight.FOUR.value if face.roll_angle else 0 
    score -= (abs(face.pan_angle) / 90) * Settings.Weight.FOUR.value if face.pan_angle else 0 
    score -= (abs(face.tilt_angle) / 90) * Settings.Weight.FOUR.value if face.tilt_angle else 0 
    return score

 
def score_pic(faces):
    score = [face_score(face) for face in faces]
    return sum(score) / len(score) if score else 0
 
 
def captureImages(queue, number_of_images=0, time_limit = 0):
    global PRINT_LOCK, FPS, client, img_counter, start
    img_counter = 0
    if time_limit: number_of_images = time_limit // FPS
 
    url = Settings.ip_cam_url
  
    while True:
        imgResp=urllib.request.urlopen(url).read()
        queue.put(imgResp)
        img_counter+=1
        if img_counter >= number_of_images: break
        time.sleep(FPS)
 
 
def threader(queue, img_dict):
    global PRINT_LOCK, FPS, client, img_counter, start
    while True:
        if not queue.empty():
            img = queue.get()
            image_content, faces = detect_faces(img)
            if faces: faces = get_valid_faces(get_faces_data(faces))
            score = score_pic(faces)
            result = check_smile(faces)
            if result and score: img_dict[score] = image_content
            with PRINT_LOCK:
                if score: print(f'Valid faces detected = {len(faces)},\tScore = {score},\tResult = {result}')
 

def get_brightness(image):
   stat = ImageStat.Stat(image)
   return stat.rms[0]


def change_brightness(image, factor):
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(factor)
    return image_brightened, get_brightness(image_brightened)


def finalize_image(image):
    try:
        score, content = image
        image = Image.open(io.BytesIO(content))

        image_brightness = get_brightness(image)
        factor = Settings.enhancment_median / ((image_brightness - Settings.enhancment_median) / 2.5 + Settings.enhancment_median)
        image, new_brightness = change_brightness(image, factor)
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = Settings.sharpness_factor
        image_sharped = enh_sha.enhance(sharpness) 
        image_sharped.save(f"{Settings.save_path}/image_{score}.jpg")
    except Exception as e:
        print(e)


def main():
    wx_app = wx.App(0)
    wx_app.MainLoop()
    pic_message = wx.BusyInfo("Taking pictures...")
    queue = Queue()
    img_dict = dict()
 
    try: shutil.rmtree(Settings.save_path)
    except FileNotFoundError: pass
    finally: os.makedirs(Settings.save_path)
    
    captureImages_thread = threading.Thread(target=captureImages, args=(queue, 0, Settings.seconds_to_run))
    captureImages_thread.daemon = True
    captureImages_thread.start()
 
    for process in range(Settings.number_of_processes):
        thread = threading.Thread(target=threader, args=(queue, img_dict))
        thread.daemon = True
        thread.start()
 
    queue.join()
    captureImages_thread.join()

    del pic_message

    processing_message = wx.BusyInfo("Processing pictures...")

    img_dict = OrderedDict(sorted(img_dict.items(), reverse=True)[:Settings.max_pics_saved])


    with cf.ProcessPoolExecutor(Settings.number_of_processes) as ex:
        ex.map(finalize_image, img_dict.items())

    if len(img_dict.keys()):
        max_score = max(img_dict.keys())
        final_image = Image.open(f"{Settings.save_path}/image_{max_score}.jpg")
        write_to_image(max_score, final_image, 0)
        for score_status, score_range in Settings.score_ranges.items():
            if int(max_score * 1000) in score_range:
                write_to_image(score_status, final_image, 100)
                break
        del processing_message
        final_image.show()

    del wx_app


if __name__ == '__main__':
    global PRINT_LOCK, FPS, client, img_counter, start, BLUR_THRESHOLD
    PRINT_LOCK = threading.Lock()
    BLUR_THRESHOLD = Settings.blur_threshold
    FPS = Settings.frames_per_second
    CREDNTIALS = Settings.json_path

    client = vision.ImageAnnotatorClient.from_service_account_json(CREDNTIALS)
    img_counter = 0
    start = 0
    main()