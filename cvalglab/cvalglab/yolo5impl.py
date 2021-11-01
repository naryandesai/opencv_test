import cv2
import numpy as np
import glob
import imutils
import imagefilters
import datetime
import os, shutil, sys
import math
from collections import defaultdict
from numba import jit
import threading
from enum import Enum
import random
import json


    

def convert_bdd_train_labels(dest_folder, maxcount):
    def transform(item, stats, source_image_path, dest_label_path, dest_image_path):
        def extract_filename_from_path(fn):
            fn = os.path.basename(fn)
            fn = os.path.splitext(fn)[0]
            return fn

        if not hasattr(transform, "category_map"):
            transform.category_map = {
                "pedestrian":0,
                "rider":1, 
                "car":2,
                "truck":3,
                "bus":4,
                "motorcycle":5,
                "bicycle":6,
                "traffic light":7,
                "traffic sign":8,

                "other vehicle": 9,
                "other person": 10,
                "trailer": 11,
                "train": 12 }

        image_name = item["name"]
        source_image_fn = os.path.join(source_image_path, image_name)
        label_content = ""
        if os.path.isfile(source_image_fn):
            source_image = cv2.imread(source_image_fn)
            height,width,_ = source_image.shape
            iheight = float(height)
            iwidth = float(width)

            if "labels" in item:
                elements = item["labels"]
                for item in elements:
                    category = item["category"]
                    category_number = 13 # unknown category
                    if category in transform.category_map:
                        category_number = transform.category_map[category]
                    if category_number <= 8:
                        raw_box = item["box2d"]
                   
                        fbx1 = float(raw_box["x1"])
                        fby1 = float(raw_box["y1"])
                        fbx2 = float(raw_box["x2"])
                        fby2 = float(raw_box["y2"])

                        xmin = min(fbx1,fbx2)
                        xmax = max(fbx1,fbx2)
                        ymin = min(fby1,fby2)
                        ymax = max(fby1,fby2)
                        
                        center_x = (xmin + xmax) / (2.0 * iwidth)
                        center_y = (ymin + ymax) / (2.0 * iheight)
                        box_width = abs(xmax - xmin) / iwidth
                        box_height = abs(ymax - ymin) / iheight

                        current_count = stats[category]
                        current_count += 1
                        stats[category] = current_count
                        
                        label_content += "{0} {1:.9f} {2:.9f} {3:.9f} {4:.9f}\n".format(category_number,center_x,center_y,box_width,box_height)

                txt_name = extract_filename_from_path(source_image_fn)
                txt_filename = os.path.join(dest_label_path,txt_name+".txt")

                with open(txt_filename, "wt") as txt_file:
                    txt_file.write(label_content)
                dest_image_fn = os.path.join(dest_image_path, image_name)
                shutil.copy(source_image_fn, dest_image_fn)
                print(f"{dest_image_fn} : {txt_filename} -- converted.")
        else:
            print(f"{source_image_fn} -- file not found")
            

    train_labels_path = "D:/bdd/bdd100k/labels/det_20/det_train.json"
    valid_labels_path = "D:/bdd/bdd100k/labels/det_20/det_val.json"
    train_image_path = "D:/bdd/bdd100k/images/100k/train"
    valid_image_path = "D:/bdd/bdd100k/images/100k/val"

    dest_train_labels = os.path.join(dest_folder, "Labels", "Train")
    dest_train_images = os.path.join(dest_folder, "Images", "Train")
    dest_val_labels = os.path.join(dest_folder, "Labels", "Valid")
    dest_val_images = os.path.join(dest_folder, "Images", "Valid")

    os.mkdir(os.path.join(dest_folder, "Labels"))
    os.mkdir(os.path.join(dest_folder, "Images"))
    os.mkdir(dest_train_labels)
    os.mkdir(dest_train_images)
    os.mkdir(dest_val_images)
    os.mkdir(dest_val_labels)

    raw_training_labels = None
    raw_val_labels = None

    print("loading training labels")
    with open(train_labels_path, "r") as load_file:
        raw_training_labels = json.load(load_file)
    print("loading validation labels")
    with open(valid_labels_path, "r") as load_file:
        raw_val_labels = json.load(load_file)

    if maxcount <= 0:
        maxcount = sys.maxsize

    catstats = defaultdict(lambda: 0)
    count = 1
    total = len(raw_training_labels)
    print(f"convert training data, {total} images to process")
    
    for it in raw_training_labels:
        print(f"{count}/{total}:", end='')
        transform(it, catstats, train_image_path, dest_train_labels, dest_train_images)
        count += 1
        if count >= maxcount:
            break
    
    valstats = defaultdict(lambda: 0)
    count = 1
    total = len(raw_val_labels)
    print(f"convert validation data, {total} images to process")
    for it in raw_val_labels:
        print(f"{count}/{total}:", end='')
        transform(it, valstats, valid_image_path, dest_val_labels, dest_val_images)
        count += 1
        if count >= maxcount:
            break
    
    print("training stats:")
    print(json.dumps(catstats, indent=4, sort_keys=True))
    print("validation stats:")
    print(json.dumps(valstats, indent=4, sort_keys=True))


################################################################

def test_bdd_train_labels():
    def extract_filename_from_path(fn):
            fn = os.path.basename(fn)
            fn = os.path.splitext(fn)[0]
            return fn

    if not hasattr(test_bdd_train_labels, "category_map"):
        test_bdd_train_labels.category_map = {
            0: "pedestrian",
            1: "rider", 
            2: "car",
            3: "truck",
            4: "bus",
            5: "motorcycle",
            6: "bicycle",
            7: "traffic light",
            8: "traffic sign" }
    
    train_labels_path = "G://code/datasets/adas01/labels/train"
    train_image_path = "G://code/datasets/adas01/images/train/*.jpg"
    image_paths = glob.glob(train_image_path)
    random.shuffle(image_paths)
    blue = (255,0,0)
    for image_path in image_paths:
        base_name = extract_filename_from_path(image_path)
        labels_name = os.path.join(train_labels_path, base_name + ".txt")
        if os.path.isfile(labels_name):
            source_image = cv2.imread(image_path)
            height, width, channels = source_image.shape
            with open(labels_name) as f:
                lines = [line.rstrip() for line in f]
                for line in lines:
                    tokens = line.split()
                    if len(tokens) == 5:
                        category_num = int(tokens[0])
                        category = test_bdd_train_labels.category_map[category_num]
                        center_x = tokens[1]
                        center_y = tokens[2]
                        box_width = tokens[3]
                        box_height = tokens[4]
                        rescaled_center_x = (int) (float(center_x) * float(width))
                        rescaled_center_y = (int) (float(center_y) * float(height))
                        rescaled_width = (int) (float(box_width) * float(width))
                        rescaled_height = (int) (float(box_height) * float(height))
                        half_width = rescaled_width // 2
                        half_height = rescaled_height // 2

                        xmin = rescaled_center_x - half_width
                        xmax = rescaled_center_x + half_width
                        ymin = rescaled_center_y - half_height
                        ymax = rescaled_center_y + half_height
                        v0 = (xmin, ymin)
                        v1 = (xmax, ymax)

                        source_image = cv2.rectangle(source_image, v0,v1, blue, 2)
                        imagefilters.put_text(img=source_image,text=category,org=v0,font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            cv2.imshow("image", source_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()





if __name__ == '__main__':
    print("1. convert bdd training data to yolo v5 format")
    print("2. convert bdd training data to yolo v5 format test run")
    print("3. visualize label data")
    choice = input("enter choice:")
    if choice == "1":
        convert_bdd_train_labels("G:/code/datasets/adas01",-1)
    elif choice == "2":
        convert_bdd_train_labels("G:/code/datasets/adas01", 100)
    elif choice == "3":
        test_bdd_train_labels()



