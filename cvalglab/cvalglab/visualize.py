from keras.layers.core import Dropout
from numpy.lib.function_base import quantile
from homegrownvp import NietoVPDetection
from homegrownvp import SegmentVPDetection
import cv2
import numpy as np
from vp_detection import VPDetection
import glob
import imutils
import imagefilters
import datetime
import os
import math
from collections import defaultdict
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import feature
import h5py
from numba import jit
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
from joblib import dump, load
import threading
from enum import Enum
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
import tensorflow as tf
...


class BDDCategories:
    road = 0
    sidewalk = 1
    building = 2
    wall = 3
    fence = 4
    pole = 5
    traffic_light = 6
    traffic_sign = 7
    vegetation = 8
    terrain = 9
    sky = 10
    person = 11
    rider = 12
    car = 13
    truck = 14
    bus = 15
    train = 16
    motorcycle = 17
    bicycle = 18
    ignore = 255

    scannables = [traffic_light,traffic_sign,person,rider,car,truck,bus,train,motorcycle,bicycle]

class GeometrisCategories:
    road = 1
    scannable = 2
    ignore = 3

trainData = None
testData = None
trainLabels = None
testLabels = None
    

def visualize_vpA(filepath):
    image_paths = glob.glob(filepath)
    for file in image_paths:
        original = cv2.imread(file)
        oHeight, oWidth, channels = original.shape

        # cropped = original[oHeight//2:oHeight,:]
        gsimg = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        twidth = 500
        gsimg = imutils.resize(gsimg, width = twidth)
        length_thresh = twidth // 20
        seed = 10101

        vpd = VPDetection(length_thresh, seed = seed)
        vps = vpd.find_vps(gsimg)

        print("------------------------")
        print(file)
        print("3d vanishing points:")
        print(vps)
        print("image vanishing points:")
        vp2d = vpd.vps_2D
        print(vp2d)
        print("------------------------")

        dbg = vpd.create_debug_VP_image()
        dbg = cv2.circle(dbg, vp2d[0].astype(int), 5, (255,0,0), 5)
        dbg = cv2.circle(dbg, vp2d[1].astype(int), 5, (0,255,0), 5)
        dbg = cv2.circle(dbg, vp2d[1].astype(int), 5, (0,0,255), 5)
        cv2.imshow(file, dbg)
        cv2.waitKey(0)
        cv2.destroyWindow(file)

def visualize_sr_filter(filepath):
    image_paths = glob.glob(filepath)
    width_sm = 30
    width_med =40
    width_lg = 50
    
    thr_sm = 80
    thr_med = 100
    thr_lg = 120

    redo = True
    while redo:

        for file in image_paths:
            
            original = cv2.imread(file)
            oHeight, oWidth, _ = original.shape
            twidth = 800

            rsimg = imutils.resize(original, width = twidth)
            gsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2GRAY)
            inputImage = rsimg
            
    
                    
            scanneda0 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_lg, thr_sm)
            imagefilters.put_text(img=scanneda0,text=f"w/w:{width_lg}. t:{thr_sm}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scanneda1 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_lg, thr_sm)
            imagefilters.put_textput_text(img=scanneda1,text=f"w/w:{width_med}. t:{thr_sm}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scanneda2 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_sm, thr_sm)
            imagefilters.put_textput_text(img=scanneda2,text=f"w/w:{width_sm}. t:{thr_sm}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
        
            scannedb0 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_lg, thr_med)
            imagefilters.put_textput_text(img=scannedb0,text=f"w/w:{width_lg}. t:{thr_med}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scannedb1 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_med, thr_med)
            imagefilters.put_textput_text(img=scannedb1,text=f"w/w:{width_med}. t:{thr_med}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scannedb2 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_sm, thr_med)
            imagefilters.put_textput_text(img=scannedb2,text=f"w/w:{width_sm}. t:{thr_med}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
        
            scannedc0 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_lg, thr_lg)
            imagefilters.put_textput_text(img=scannedc0,text=f"w/w:{width_lg}. t:{thr_lg}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scannedc1 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_med, thr_lg)
            imagefilters.put_textput_text(img=scannedc1,text=f"w/w:{width_med}. t:{thr_lg}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
            scannedc2 = imagefilters.BoostedStructuredRoadFilter(inputImage, twidth//width_sm, thr_lg)
            imagefilters.put_textput_text(img=scannedc2,text=f"w/w:{width_sm}. t:{thr_lg}",org=(5,35),font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, thickness=2, font_scale=1.5, color=(255,255,255))
        
            comparisona = cv2.hconcat([scanneda0,scanneda1,scanneda2])
            comparisonb = cv2.hconcat([scannedb0,scannedb1,scannedb2])
            comparisonc = cv2.hconcat([scannedc0,scannedc1,scannedc2])
        
            comparison = cv2.vconcat([comparisona,comparisonb,comparisonc])
            

            cv2.imshow("original", rsimg)
            cv2.imshow(file, comparison)

            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
        
        choice = input("input parameters (y/n)?")
        redo = choice == "y"
        if redo:
            print(f"width: sm:{width_sm}, md:{width_med}, lg:{width_lg}")
            print(f"thr: sm:{thr_sm}, md:{thr_med}, lg:{thr_lg}")
            ip = input("width small")
            width_sm = int(ip)
            ip = input("width med")
            width_med = int(ip)
            ip = input("width large")
            width_lg = int(ip)
            ip = input("threshold small")
            thr_sm = int(ip)
            ip = input("threshold med")
            thr_med = int(ip)
            ip = input("threshold large")
            thr_lg = int(ip)

def visualize_nieto_vp(filepath, lowerOnly = False):
    image_paths = glob.glob(filepath)
    for file in image_paths:
        
        original = cv2.imread(file)
        oHeight, oWidth, _ = original.shape
        twidth = oWidth
        
        rsimg = imutils.resize(original, width = twidth)
       
        vpfinder = NietoVPDetection(debugMode=True)

        dbg,dbgB,fimg = vpfinder.find_forward_vanishing_point(rsimg,lowerHalf=lowerOnly)
        
        comparison = cv2.hconcat([rsimg,dbg])
        comparison = cv2.vconcat([comparison, cv2.hconcat([dbgB,fimg])])
        cv2.imshow(file, comparison)

        cv2.waitKey(0)
        cv2.destroyWindow(file)

def visualize_seg_vp(filepath, lowerOnly = False):
    image_paths = glob.glob(filepath)
    for file in image_paths:
        
        original = cv2.imread(file)
        
        rsimg = imutils.resize(original, width = 512)
       
        vpfinder = SegmentVPDetection(debugMode=True)

        dbg,dbgB,fimg = vpfinder.find_forward_vanishing_point(rsimg,lowerHalf=lowerOnly)
        
        comparison = cv2.hconcat([rsimg,dbg])
        comparison = cv2.vconcat([comparison, cv2.hconcat([dbgB,fimg])])
        cv2.imshow(file, comparison)

        cv2.waitKey(0)
        cv2.destroyWindow(file)

def visualize_nieto_improvement(filepath):
    image_paths = glob.glob(filepath)
    image_paths.reverse()
    

    for file in image_paths:
        
        original = cv2.imread(file)
        oHeight, oWidth, _ = original.shape
        twidth = 800

        rsimg = imutils.resize(original, width = twidth)
        inputImage = rsimg
        
        scanwidth = twidth//25
        scanned0 = imagefilters.NietoStructuredRoadFilter(inputImage, scanwidth, 90, start_row_pct=0.5, end_row_pct=0.8)
        scanned0cvt = cv2.cvtColor(scanned0, cv2.COLOR_GRAY2BGR)
        cv2.line(scanned0cvt, (10,10),(10+scanwidth,10), (255,0,0), 2)
        scanned1, pp = imagefilters.BoostedStructuredRoadFilter(inputImage, scanwidth, 90, start_row_pct=0.5, end_row_pct=0.8)
        scanned1cvt = cv2.cvtColor(scanned1, cv2.COLOR_GRAY2BGR)
        cv2.line(scanned1cvt, (10,10),(10 + scanwidth, 10), (255,0,0), 2)
        
        comparisona = cv2.hconcat([inputImage, scanned0cvt])
        comparisonb = cv2.hconcat([pp, scanned1cvt])
        comparison = cv2.vconcat([comparisona,comparisonb])
        cv2.imshow(file, comparison)

        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

def video_pedestrian(filepath):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    video = cv2.VideoCapture(filepath)

    cv2.namedWindow("pedestrians", cv2.WINDOW_AUTOSIZE)

    while video.isOpened():
        ret,frame = video.read()

        if ret == True:
            frame = imutils.resize(frame,width=min(800,frame.shape[1]))

            start = datetime.datetime.now()
            ppframe = imagefilters.white_balance(frame)
            (rects, weights) = hog.detectMultiScale(ppframe, winStride=(4,6), padding=(8,8), scale=1.05, useMeanshiftGrouping=False)
            print("[INFO] detection took: {}s".format((datetime.datetime.now() - start).total_seconds()))

            for (rect, w) in zip(rects,weights):
                (x, y, w, h) = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imagefilters.put_text(frame, str(w), (x+5, y+15), font_face=cv2.FONT_HERSHEY_SIMPLEX, thickness=2, font_scale=0.5, color=(0,0,255))

            cv2.imshow("pedestrians", frame)

            # define q as the exit button
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    video.release()
    
    cv2.destroyAllWindows()

def visualize_ppl_detect(which):
    if which == 1:
        video_pedestrian("pedestrians/01.mp4")
    elif which == 2:
        video_pedestrian("pedestrians/02.mp4")
    elif which == 3:
        video_pedestrian("pedestrians/03.mp4")
    
def total_milliseconds(start):
    seconds = (datetime.datetime.now() - start).total_seconds()
    ms = int(seconds * 1000.0)
    return ms

def visualize_segmentation(filepath):
    image_paths = glob.glob(filepath)

    sharpen_filter = np.array(
        [
            [0, -1, 0], 
            [-1, 5, -1], 
            [0, -1, 0]])

    stdw = 320

    for file in image_paths:
        
        original = io.imread(file)
        
        working = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
        contrast = cv2.convertScaleAbs(working,alpha=1.4)
        sharp = cv2.filter2D(contrast, -1, sharpen_filter)
        sharp = cv2.filter2D(sharp, -1, sharpen_filter)
        wb = imagefilters.white_balance(sharp)
                
        

        original = imutils.resize(original, width=stdw)
        inputimg = imutils.resize(wb, width=stdw)

        img = img_as_float(inputimg)
        pp = cv2.cvtColor(inputimg, cv2.COLOR_RGB2BGR)
        oimg = img_as_float(pp)
        
        start = datetime.datetime.now()
        fscale = int(0.8 * stdw) # int(0.618 * stdw)
        segments_fz = felzenszwalb(img, scale=fscale, sigma=0.5, min_size=fscale//5) 
        print("Felzenswalb took: {}ms".format(total_milliseconds(start)))
        print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")

        start = datetime.datetime.now()
        gradient = sobel(rgb2gray(img))
        segments_watershed = watershed(gradient, markers=180, compactness=0.001) # markers 250
        print("Watershed took: {}ms".format(total_milliseconds(start)))
        print(f"Watershed number of segments: {len(np.unique(segments_watershed))}")
        

        fig, ax = plt.subplots(2, 2, figsize=(8,8), sharex=True, sharey=True)
        ax[0, 0].imshow(imutils.resize(mark_boundaries(oimg, segments_fz), width=stdw))
        ax[0, 0].set_title("Felzenszwalbs's method")
        
        ax[0, 1].imshow(imutils.resize(mark_boundaries(oimg, segments_watershed), width=stdw))
        ax[0, 1].set_title('Compact watershed')

        ax[1, 0].imshow(original)
        ax[1, 0].set_title('Original')
        
        ax[1, 1].imshow(pp)
        ax[1, 1].set_title('Preprocessed')

        for a in ax.ravel():
            a.set_axis_off()

        # plt.tight_layout()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()
        
        cv2.waitKey(0)

        plt.close('all')
        
def extract_filename_from_path(fn):
    fn = os.path.basename(fn)
    fn = os.path.splitext(fn)[0]
    return fn

def extract_image_feature_data(labimage, labels, lbp, truth):
    rows,cols = labels.shape

    region_colors = defaultdict(list)
    region_textures = defaultdict(list)
    region_locations = defaultdict(list)
    region_truths = defaultdict(list)
    
    for row in range (0, rows):
        for col in range(0, cols):
            seg_label = labels[row, col]
            raw = truth[row,col]
            truth_category = int(raw)  

            if truth_category == BDDCategories.road:
                truth_category = GeometrisCategories.road
            elif truth_category in BDDCategories.scannables:
                truth_category = GeometrisCategories.scannable
            else: 
                truth_category = GeometrisCategories.ignore

            l,a,b = labimage[row, col]
            texture = lbp[row, col]

            region_colors[seg_label].append((l,a,b))
            region_textures[seg_label].append(texture)
            region_locations[seg_label].append((row,col))
            region_truths[seg_label].append(truth_category)

    return (region_colors,region_textures,region_locations,region_truths)

def normalize_histogram(hist):
    eps = 1e-7
    retval = hist.astype(np.float32)
    retval /= (hist.sum() + eps)
    return retval

def create_region_feature(region_index, colors, textures, locations, truth, lbppoints, height, width):
    rcolors = colors[region_index]
    rtextures = textures[region_index]
    rlocations = locations[region_index]
    rtruths = truth[region_index]
    
    truth_counts = np.bincount(rtruths,minlength = 255)
    best_truth = float(np.argmax(truth_counts))


    center = [sum(ele) / len(rlocations) for ele in zip(*rlocations)]
    minp = list(map(min, zip(*rlocations)))
    maxp = list(map(max, zip(*rlocations)))

    exta = abs(maxp[0]-minp[0]) / height
    extb = abs(maxp[1]-minp[1]) / width
    area = len(rlocations) / (height * width)

    cy,cx = center
    
    highest = minp[0] / height
    
    placement = [cx/width, cy/height, area, exta, extb, highest]
    

    BinCount = 64
    lum = np.array([c[0] for c in rcolors])
    ca = np.array([c[1] for c in rcolors])
    cb = np.array([c[2] for c in rcolors])

    histogram_l = normalize_histogram(np.histogram(lum, bins=BinCount, range=(0,256))[0])
    histogram_a = normalize_histogram(np.histogram(ca, bins=BinCount, range=(0,256))[0])
    histogram_b = normalize_histogram(np.histogram(cb, bins=BinCount, range=(0,256))[0])
    histogram_t = normalize_histogram(np.histogram(rtextures, bins=range(0,lbppoints+3), range=(0,lbppoints+2))[0])

    
    f = np.concatenate((placement, histogram_l, histogram_a, histogram_b, histogram_t))
    return f, best_truth

def create_segment_descriptors():
    raw_images_path = "D:/bdd/bdd100k/images/10k/train/*.jpg"
    seg_images_path = "D:/bdd/bdd100k/labels/sem_seg/masks/train/*.png"

    all_training_image_paths = glob.glob(raw_images_path)
    all_seg_image_paths = glob.glob(seg_images_path)
    seg_name_lookup = dict()
    for sp in all_seg_image_paths:
        sn = extract_filename_from_path(sp)
        seg_name_lookup[sn] =sp

    ImageWidth = 320
    sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    lbppoints = 8 # 8 up to 24
    lbpradius = 1 # 1 up to 3
    # fscale = int(0.8 * ImageWidth) # 
    fscale = int(0.618 * ImageWidth)


    with h5py.File('segmentfeatures.hdf5', 'w') as feature_file:
        frow = 0
        image_truth = []

        for imgpath in all_training_image_paths:
            imname = extract_filename_from_path(imgpath)
            if imname not in seg_name_lookup:
                continue

            segpath = seg_name_lookup[imname]
                
            original = imutils.resize(io.imread(imgpath), width=ImageWidth)
            height, width, channels = original.shape
            truth_color_map = io.imread(segpath)
            ground_truth = imutils.resize(truth_color_map, width=ImageWidth, inter=cv2.INTER_NEAREST) 
            

            start = datetime.datetime.now()
            working = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            contrast = cv2.convertScaleAbs(working,alpha=1.4)
            sharp = cv2.filter2D(contrast, -1, sharpen_filter)
            sharp = cv2.filter2D(sharp, -1, sharpen_filter)
            input_image = imagefilters.white_balance(sharp)
            lab_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)
            gs_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            img = img_as_float(input_image)
            
            #labels = felzenszwalb(img, scale=fscale, sigma=0.5, min_size=fscale//5).astype(np.int32)
            gradient = sobel(rgb2gray(img))
            labels = watershed(gradient, markers=180, compactness=0.001) # markers 250

            lbp = feature.local_binary_pattern(gs_image, P=lbppoints, R = lbpradius, method="uniform")
            
            un_labels = np.unique(labels)
            
            (rcolors, rtextures, rlocations, rtruth) = extract_image_feature_data(lab_image,labels, lbp,ground_truth)
            # print(f'processing {len(un_labels)} labels for image {imname}')
            image_features = []
            flen = 0

            for region in un_labels:
                f, t = create_region_feature(region, rcolors, rtextures, rlocations, rtruth, lbppoints, height, width)
                flen = len(f)
                image_features.append(f)
                image_truth.append(t)
            
            features_data = np.asarray(image_features)
            analysis_time = total_milliseconds(start)

            if frow == 0:
                feature_file.create_dataset('features', (len(image_features), flen), chunks=True, data=features_data, maxshape=(None,flen))
            else:
                feature_file['features'].resize((feature_file['features'].shape[0] + features_data.shape[0]),axis=0)
                feature_file['features'][-features_data.shape[0]:] = features_data
            
            
            fshape = feature_file['features'].shape
            print(f'processed iteration {frow} / {len(all_training_image_paths)}. {len(un_labels)} regions in {analysis_time} milliseconds. data shape {fshape}')
            frow += 1   
            
        feature_file.create_dataset('labels', chunks=True, data=image_truth)

def getsegdata():
    feature_file = h5py.File('balancedsegmentfeatures.hdf5', 'r')
    feature_data = np.asarray(feature_file['features'])
    label_data = np.asarray(feature_file['labels'])
    feature_file.close
    global trainData
    global testData
    global trainLabels
    global testLabels
    trainData, testData, trainLabels, testLabels = train_test_split(feature_data, label_data, test_size=0.25, random_state=987)

   


def trainmodel(model, title, fname):
    global trainData
    global testData
    global trainLabels
    global testLabels
    print("training " + title)
    model.fit(trainData, trainLabels)
    report = classification_report(testLabels, model.predict(testData))
    print("-------------------------------------")
    print(title + ' report:')
    print(report)
    print("-------------------------------------")
    print("")
    dump(model, fname)

def train_knn():
    trainmodel(KNeighborsClassifier(5), "k nearest neighbors:", 'segknn.joblib')

def trainnusvc():
    trainmodel(NuSVC(probability=True), "nu svc:", 'segnusvc.joblib')

def traindectree():
    trainmodel(DecisionTreeClassifier(), "decision tree", 'segdectree.joblib')    

def trainadaboost():
    trainmodel(AdaBoostClassifier(), "ada boost", 'segadaboost.joblib')

def traingradboost():
    trainmodel(GradientBoostingClassifier(), "gradient boosting", 'seggradboost.joblib')

def traingaussiannb():
    trainmodel(GaussianNB(),"gaussian nb", 'seggaussiannb.joblib')

def trainlindiscr():
    trainmodel(LinearDiscriminantAnalysis(), "linear discriminant analysis", 'seglindisc.joblib')

def trainquaddiscr():
    trainmodel(QuadraticDiscriminantAnalysis(), "quadratic discriminante analysis", 'segquaddisc.joblib')

def train_seg_poly_svm():
    
    model = SVC(kernel="rbf", C=0.025, probability=True, max_iter=1e4)
    trainmodel(model,"poly svm classifier",'segpolysvm.joblib')

def train_seg_sgd_svm():
    feature_file = h5py.File('balancedsegmentfeatures.hdf5', 'r')
    feature_data = np.asarray(feature_file['features'])
    label_data = np.asarray(feature_file['labels'])
    feature_file.close
    (trainData, testData, trainLabels, testLabels) = train_test_split(feature_data, label_data, test_size=0.25, random_state=987)

    

    scaler = StandardScaler()
    scaler.fit(trainData)
    scaledTrainData = scaler.transform(trainData)
    scaledTestData = scaler.transform(testData)
    model = SGDClassifier(loss='hinge', penalty="l2", max_iter=1e6, tol=1e-5, n_jobs=4, random_state=789)
    print(f'training sgd svm, starting at ... {datetime.datetime.now() }')
    model.fit(scaledTrainData, trainLabels)
    report = classification_report(testLabels, model.predict(scaledTestData))
    print("sgd svm classifier report:")
    print(report)
    print("")
    dump(model, 'segsgdsvm.joblib')

def train_seg_rand_forest():
    model = RandomForestClassifier(n_estimators=400, random_state=123, n_jobs=6)
    trainmodel(model, "random forest classifier", 'segrforest.joblib')


    

def train_seg_ann():
    feature_file = h5py.File('balancedsegmentfeatures.hdf5', 'r')
    feature_data = np.asarray(feature_file['features'])
    label_data = np.asarray(feature_file['labels'])
    feature_file.close
    encoder = LabelEncoder()
    encoder.fit(label_data)
    encoded_labels = encoder.transform(label_data)
    onehotlabels = np_utils.to_categorical(encoded_labels)
    
    model = Sequential()
    model.add(Dropout(0.25, input_shape=(208,)))
    
    model.add(Dense(int(208), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(int(208 * 5), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    trainData, testData, trainLabels, testLabels = train_test_split(feature_data, onehotlabels, test_size=0.25, random_state=789)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(trainData, trainLabels, epochs=200, batch_size=20, validation_data=(testData,testLabels), callbacks=[tensorboard_callback])
    model.save('segann.h5')
   


def train_seg_models():
    getsegdata()

    t0 = threading.Thread(target = train_seg_poly_svm)
    t1 = threading.Thread(target = train_seg_sgd_svm)
    t2 = threading.Thread(target = train_seg_rand_forest)
    t3 = threading.Thread(target = train_knn)
    t4 = threading.Thread(target = trainnusvc)
    t5 = threading.Thread(target = traindectree)
    t6 = threading.Thread(target = trainadaboost)
    t7 = threading.Thread(target = traingradboost)
    t9 = threading.Thread(target= traingaussiannb)
    t10 = threading.Thread(target=trainlindiscr)
    t11 = threading.Thread(target = trainquaddiscr)

    threads = [t0,t1,t2,t3,t4,t5,t6,t7,t9,t10,t11]
    for th in threads:
        th.start()

    for th in threads:
        th.join()
   

    print("Training completed.")

def balance_training_data():
    feature_file = h5py.File('segmentfeatures.hdf5', 'r')
    feature_data = np.asarray(feature_file['features'])
    label_data = np.asarray(feature_file['labels'])
    feature_file.close
    ixlabels_road = [i for i,x in enumerate(label_data) if x==GeometrisCategories.road]
    random.shuffle(ixlabels_road)
    ixlabels_scan = [i for i,x in enumerate(label_data) if x==GeometrisCategories.scannable]
    random.shuffle(ixlabels_scan)
    ixlabels_ignore = [i for i,x in enumerate(label_data) if x==GeometrisCategories.ignore]
    random.shuffle(ixlabels_ignore)

    balanced_len = min(min(len(ixlabels_road), len(ixlabels_scan)), len(ixlabels_ignore))
    ixlabels_road = ixlabels_road[:balanced_len]
    ixlabels_scan = ixlabels_scan[:balanced_len]
    ixlabels_ignore = ixlabels_ignore[:balanced_len]
    
    all_indices = ixlabels_road + ixlabels_scan + ixlabels_ignore
    random.shuffle(all_indices)

    feature_data = feature_data[all_indices]
    label_data = label_data[all_indices]
    
    with h5py.File('balancedsegmentfeatures.hdf5', 'w') as balanced_file:
        balanced_file.create_dataset('features', chunks=True, data=feature_data)
        balanced_file.create_dataset('labels', chunks=True, data=label_data)




if __name__ == '__main__':
    print("select test set")
    print("1. roadway general")
    print("2. roadway demo")
    print("3. roadway edge cases")
    print("4. training 10k")
    choice = input("enter choice:")
    folder = "vptest"
    if choice == "1":
        folder = "vptest"
    elif choice == "2":
        folder = "rwdemo"
    elif choice == "3":
        folder = "rwedge"
    elif choice == "4":
        folder = "D:/bdd/bdd100k/images/10k/train"
    folder += "/*.jpg"

    print("1: Boosted Structured road image filter")
    print("2: Plain vs Boosted comparison")
    print("3: Xiaohu Lu VP Detection")
    print("4: Boosted Nieto VP Detection")
    print("5: Boosted Lower Nieto VP Detection")
    print("6: Pedestrian detection")
    print("7: Image segmentation")
    print("8: segmentation vp detection")
    print("9: segmentation lower vp detection")
    print("10: create segmentation features")
    print("11: train segmentation features models")
    print("12: balance feature data")
    print("13: train segmentation features ANN ")

    choice = input("enter choice:")
    print(f"running option {choice}")
    if choice == "3":
        visualize_vpA(folder)
    elif choice == "2":
        visualize_nieto_improvement(folder)
    elif choice == "1":
        visualize_sr_filter(folder)
    elif choice == "4":
        visualize_nieto_vp(folder,False)
    elif choice == "5":
        visualize_nieto_vp(folder,True)
    elif choice == "6":
        print("pedestrian vid choice (1-3)")
        vid = input("video number:")
        visualize_ppl_detect(int(vid))
    elif choice == "7":
        visualize_segmentation(folder)
    elif choice == "8":
        visualize_seg_vp(folder, False)
    elif choice == "9":
        visualize_seg_vp(folder, True)
    elif choice == "10":
        create_segment_descriptors()
    elif choice == "11":
        train_seg_models()
    elif choice=="12":
        balance_training_data()
    elif choice=="13":
        train_seg_ann()


    