import os
import glob
import re
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score

class Object:
    def __init__(self):
        self.type = 0
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.xres = 0
        self.yres = 0
        
class Image:
    def __init__(self):
        self.number = ''
        self.location = ''
        self.objectList = []
        self.jointLocations = []
        self.persons = 0
        self.labelPath = ''
        self.imgPath = ''
        self.labelList = []
    
class Sequence:
    def __init__(self):
        self.imageDataList = []
        self.label = []
        self.dirName = ''

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def calculateDistance(obj1,obj2):
    return math.sqrt( pow(obj1[0]-obj2[0],2) + pow(obj1[1]-obj2[1],2))
    
def calculateCentroid(obj):
    lis = []
    lis.append((float(obj.xtop) + float(obj.xbot))/2)
    lis.append((float(obj.ytop) + float(obj.ybot))/2)
    return lis

def generate_featuremap(sequence):
    max_persons = -1
    for image in sequence.imageDataList:
        max_persons = max(max_persons,image.persons)
    feature_maps = []
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    for i in range(0,max_persons):
        feature1.append([])
        feature2.append([])
        feature3.append([])
        feature4.append([])
    print(feature1)
    for image in sequence.imageDataList:
        joint_locations = image.jointLocations
        #GET THE NUMBER OF PERSONS BY NUMBER OF ELEMENTS IN JOINT_LOCATIONS
        prev_bag = 0
        for i in range(0,image.persons):
            features = []
            #person_joints = joint_locations[i]
            shoulder_pos_l = [0,0]  #person_joints[x]
            wrist_pos_l = [0,0] #person_joints[x1]
            shoulder_pos_r = [0,0] #person_joints[x]
            wrist_pos_r = [0,0] #person_joints[x1]
            #feature_1 shoulder wrist distance
            features.append(calculateDistance(wrist_pos_l, shoulder_pos_l))
            features.append(calculateDistance(wrist_pos_r, shoulder_pos_r))
            feature1[i].extend(features)
                
            
            
            #feature_2 wrist - items,3 head - items ,4 bag - items  
            features2 = []
            features3 = []
            features4 = []
            count = 9  # max number of objects in an image(parameter)
            #assuming that the joint locations are given from left to right as of now.
            
            head_loc = [0,0] #person_joints[x]
            objectlist = image.objectList
            bag = Object()
            
            # finding the bag
            for it in range(prev_bag,len(objectlist)):
                 if objectlist[it].type == 8:
                    bag = objectlist[it]
                    prev_bag = it + 1
                    break
            
            
            for obj in objectlist:
                count = count - 1
                features2.append(calculateDistance(wrist_pos_l,calculateCentroid(obj)))
                features2.append(calculateDistance(wrist_pos_r,calculateCentroid(obj)))
                features3.append(calculateDistance(head_loc,calculateCentroid(obj)))
                features4.append(calculateDistance(calculateCentroid(bag),calculateCentroid(obj)))
            while count > 0:
                features2.append(0)
                features2.append(0)
                features3.append(0)
                features4.append(0)
                count = count - 1    
            feature2[i].extend(features2)
            feature3[i].extend(features3)
            feature4[i].extend(features4)
                
    #so currently the final features are like [[dis@t1 dis@t1 ...... dis@t2 dis@t2 dis@t2 ......] [dis@t1 dis@t1 ...... dis@t2 dis@t2 dis@t2 ......]]
    f1 = np.asarray(feature1)
    f2 = np.asarray(feature2)
    f3 = np.asarray(feature3)
    f4 = np.asarray(feature4)
    
    #adding the features personwise 
    res = np.hstack((f1,f2))
    res = np.hstack((res,f3))
    res = np.hstack((res,f4))
    return res
         
def generate_data(all_sequences):
    all_features = []
    all_labels = []
    for seq in all_sequences:
        features = generate_featuremap(seq)
        print(features)
        labels = seq.label
        print(labels)
        if(len(labels) == features.shape[0]):
            for it in range(0,len(labels)):
                if features[it].size == 190: #calculated from the constant value
                    all_features.append(features[it])
                    all_labels.append(labels[it])
    return all_features,all_labels

def discretize(labels):
    newlabels=[]
    for item in labels:
        print(item)
        if item == "picking":
            newlabels.append(0)
        elif item == "placing":
            newlabels.append(1)
        elif item == "cart":
            newlabels.append(2)
        elif item == "idle":
            newlabels.append(3)
        else:
            print("ERROR when discretizing.")
    return newlabels

#saving the SVM's model

from joblib import dump
def linear_classification(features,labels):
#    try:
    data = features
    labels = np.asarray(discretize(labels))
    kf = KFold(n_splits=4,shuffle=True)
    conf_mat=np.full((4,4),0)
    scores=[]
    for train_index, test_index in kf.split(data):
        train_d = []
        train_l = []
        test_d = []
        test_l = []
        for it in range(0,len(features)):
            if it in train_index:
                train_d.append(features[it])
                train_l.append(labels[it])
            if it in test_index:
                test_d.append(features[it])
                test_l.append(labels[it])
        print("done")
        train_d = np.asarray(train_d)
        #print(train_d.shape)
        test_d = np.asarray(test_d)
        #print(test_d.shape)
        global clf
        clf = svm.LinearSVC(C=3.5)
        dump(clf, 'activity_recognition.joblib') 
        clf.fit(train_d, train_l)

        ftrain_pred=clf.predict(train_d)
        ftest_pred=clf.predict(test_d)
        cur_score=accuracy_score(test_l,ftest_pred, normalize=True)
        scores.append(cur_score)
        train_score=accuracy_score(train_l,ftrain_pred,normalize=True)
        print(train_score,cur_score)
        #for j in range(len(test_l)):
        #    if(ftest_pred[j]!=test_l[j]):
        #        print("misclassified - ",ftest[j,0]," where ",num_to_exer(ftest[j,1])," as ",num_to_exer(ftest_pred[j]))
        cur_matrix=confusion_matrix(test_l,ftest_pred)
        cur_matrix=np.asarray(cur_matrix)
        conf_mat=np.add(conf_mat, cur_matrix)
    scores=np.asarray(scores)    
    print(scores.mean())
    print(conf_mat)           
#    except:
#        print("error")     

def generate_sequence(list_of_frames):
    frame = (objects,joint_locations)
    print("The number of frames are ", len(list_of_frames))
    for frame in list_of_frames:
        
def predict_sequence(objects,persons):


def loadDir():
    all_sequences = []
    imagedir = './mod-data'
    labeldir = './labels'
    dirList = sorted_alphanumeric(glob.glob(os.path.join(imagedir, '**')))
    if len(dirList) == 0:
        print("No directories found")
        
    for directory in dirList:
        cursequence = Sequence()
        cursequence.dirName = directory
        try:
            targetlabelfile = directory + "/label.txt"
            targetlabel = open(targetlabelfile, 'r')
        except IOError:
            print("Error: can\'t find file or read data from label file", directory)
        tar_labels=[]
        for eachline in targetlabel:
            eachline = eachline.rstrip('\n')
            tar_labels.append(eachline)
        cursequence.label = tar_labels
        
        imageList = sorted_alphanumeric(glob.glob(os.path.join(directory, '*.jpg')))
        num = 0
        imgdatalist = []
        for image in imageList:
            curimage = Image()
            
            curimage.persons = len(tar_labels)  #REMOVE THIS ONCE YOU GET JOINT LOCATIONS
            curimage.path = image
            curimage.number = num
            num = num + 1  
            segments = image.split('/')
            imgname = segments[len(segments) - 1]
            imgsegments = imgname.split('.')
            imgnumber = imgsegments[0]
            labelname = imgnumber + ".txt"
            
            
            try:
                location = labeldir +'/'+ labelname
                curimage.labelPath = location
                labelFile = open(location , 'r')
                objects = []
                for line in labelFile:
                    curobject = Object()
                    input_numbers = line.split(' ')
                    if len(input_numbers) == 7:
                        curobject.type = int(input_numbers[0])
                        curobject.xtop = int(input_numbers[1])
                        curobject.ytop = int(input_numbers[2])
                        curobject.xbot = int(input_numbers[3])
                        curobject.ybot = int(input_numbers[4])
                        curobject.xres = int(input_numbers[5])
                        curobject.yres = int(input_numbers[6])
                    else:
                        print("Read ERROR len not 7")
                    objects.append(curobject)
                curimage.objectList = objects
            except IOError:
               print("Error: can\'t find file or read data")
            
            
            
            #write code to get the joint locations
            
            
            imgdatalist.append(curimage)
        cursequence.imageDataList = imgdatalist    
        all_sequences.append(cursequence)
    return all_sequences
        
all_sequences = loadDir()   
all_features, all_labels = generate_data(all_sequences)
linear_classification(all_features,all_labels)
