#import sys
#sys.path.insert(0, '/path/to/application/app/folder')

#from activity_detection import

import cv2
def splitter_module(video_path,output_location):
    # input: video file location , output directory of frames
    # ouput: none
    # task : split video into frames
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    while success: 
        success, image = vidObj.read()
        if count%6 == 0: 
            cv2.imwrite("./frames/frame%d.jpg" % count, image) 
        count += 1
    return 

#global variables
person_details = {}
inventory = []
inventory_names = {}
object_mapper = {}

class Person:
    def __init__(self):
        self.name = ""
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.jointLocations = []
        # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
        # i have taken only 18 locations according to the parse_pickle script.
        #access via 2) index location

class PersonDetails:
    def __init__(self):
        self.name = ""    
        self.pickedUpItems = set()
        self.Wallet = 0
        self.CartItems = set()

class Object:
    def __init__(self):
        self.type = 0
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.xres = 0
        self.yres = 0

class ObjectInstance:
    def __init__(self):
        self.name = ''
        self.price = ''

class ObjectInventory:
    def __init__(self):
        self.name = ''
        self.price = ''
        self.quantity = 0

class Image:
    def __init__(self):
        self.number = ''
        self.location = ''
        self.objectList = []
        self.personList = []
        self.labelPath = ''
        self.imgPath = ''
        self.labelList = []
    
class Sequence:
    def __init__(self):
        self.imageDataList = []
        self.label = []
        self.dirName = ''


def FillStoreInventory(filepath):
    # input: path
    # ouput: none
    # task : fills the iventory of store in global vars
    try:
        items = open(filepath, 'r')
    except IOError:
        print("Error: can\'t find file or read data ", filepath)
    count = 1
    for eachline in items:
            eachline = eachline.rstrip('\n')
            splitt = eachline.split(' ')
            if len(splitt) != 3:
                print("Check once for load error else ignore", eachline)
            newobject = ObjectInventory()
            newobject.name = splitt[0]
            newobject.price = int(splitt[1])
            newobject.quantity = int(splitt[2])
            inventory.append(newobject)
            inst = ObjectInstance()
            inst.name = newobject.name
            inst.price = newobject.price
            object_mapper[count] = inst
            count += 1
    return

    
def object_to_detail_mapper(object_number):
    # input: 1
    # ouput: object details
    #print("object_number ", objimg_name = "./images/" + str(cur_frame_number) + ".jpg"ect_number)
    if object_number in object_mapper.keys():
        return object_mapper[object_number]
    else:
        print(" ERROR in object mapping ", object_number)
    return

import math
import numpy as np

def calculateDistance(obj1,obj2):
    return math.sqrt( pow(obj1[0]-obj2[0],2) + pow(obj1[1]-obj2[1],2))
    
def calculateCentroid(obj):
    lis = []
    lis.append((float(obj.xtop) + float(obj.xbot))/2)
    lis.append((float(obj.ytop) + float(obj.ybot))/2)
    return lis
def get_bag_and_other_objects(objects):
    bags = []
    others = []
    for item in objects:
        if item.type == 8:
            bags.append(item)
        else:
            others.append(item)
    return bags,others

def get_closest_bag(objects,person):
    bags, others = get_bag_and_other_objects(objects)
    min_dist = 999999999
    closest_bag = Object()
    for item in bags:
        ans = calculateDistance(calculateCentroid(item), calculateCentroid(person))
        if ans < min_dist:
            min_dist = ans
            closest_bag = item
    others.append(closest_bag)
    return others


############################################################################################################################################
#LSTM BASED ACTIVITY DETECTION

def fill_in_missing_objects(modobjects):
    seenobjects= {}
    for obj in modobjects:
        if (obj.type - 1)  not in seenobjects.keys():
            seenobjects[obj.type - 1] = 1
    for i in range(8):
        if i  not in seenobjects.keys():
            nullobj = Object()
            nullobj.type= i + 1
            modobjects.append(nullobj)
            seenobjects[i] = 1
    return modobjects

def sort_obj(obj):
    newlist = sorted(obj, key=lambda x: x.type)
    return newlist

def generate_person_object_locations_prevframes(sequence):
    person_set = {}
    # person_set is a dictionary mapping person name to the list of tuples (object, person) over the past 5 frames.
    # person_set["person0"] = [(objects,person) , (objects,person) ,(objects,person) , (objects,person), (objects,person)]
    # there might be missing objects but the person is in all the frames with deafult values added in.
    idx = 0
    if len(sequence.imageDataList) != 5:
        print(" Maintain the length of sequence as 5")
        print(sequence.dirName)
    for item in sequence.imageDataList:
        idx += 1
        objects = item.objectList
        persons = item.personList
        for person in persons:
            objectsmod = get_closest_bag(objects,person)
            objectsmod = fill_in_missing_objects(objectsmod)
            if person.name in sorted(person_set.keys()):
                lis = person_set[person.name]
                last = lis[-1]
                previdx = last[0]
                while previdx != idx-1:
                    per = Person()
                    per.name = person.name
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
            else:
                lis = []
                previdx = 0
                while previdx != idx-1:
                    per = Person()
                    per.name = person.name
                    lis.append((previdx + 1,[],per))
                    previdx += 1
                lis.append((idx,objectsmod,person))
                person_set[person.name] = lis
    for item in sorted(person_set.keys()):
        lis = person_set[item]
        last = lis[-1]
        previdx = last[0]
        while previdx <5:
            per = Person()
            per.name = item
            lis.append((previdx + 1,[],per))
            previdx += 1
    
    for item in sorted(person_set.keys()):
        lis = person_set[item]
        newlis = []
        for it in lis:
            newlis.append( ( it[1] , it[2] ) )
        #print(len(newlis))
        person_set[item] = newlis
    sorted_lis = []
    for item in newlis:
        obj = item[0]
        obj_new = sort_obj(obj)
        sorted_lis.append((obj_new,item[1]))
        
    #also filters with only the bag closest to the person added to the objects corresponding to that person.
    # now for each person you have a the set of object locations and pose locations over the past 5 frames.
    return person_set


#########################################################################################################
#########################################################################################################
#THRESHOLD BASED ACTIVITY DETECTION

def check_activity(a1,a2,b,w, wloc, threshold,movementThreshold, bagThreshold = 45):
    # input : indexes of a1 are objects from 1 to 7: a1 = [.... [(distance, object class) (distance, object class) (distance, object class)] ..... ] over past 5 frames.
    # threshold is the distance between wrist and object - handthreshold.
    # movementthreshold: how distance have moved over 2 consecutive frames to check if object is moving.
    # bagthreshold : distance between the wrist and the bag to be considered as adding to the cart.
    # based on the assumption that the object is unique(no duplicates)

    # if the previous frame and the current frame object distance decreases and the distance b/w hand and object is within threshold == pickedup
    # if in the previous frame the object is within the threshol and now the oject is moving away from the hand == placing
    # if the hand is within the threshold distance from the bag placed and item is found out later from the set of picked items.
    detected_activity = []
    length = 5 #number of frames
    for loop in range(2,length):
        for ind in range(1,8):
            if a1[ind][loop][0] != -1 and a1[ind][loop - 1][0] != 1:     
                if a1[ind][loop][0] <= a1[ind][loop-1][0] and a1[ind][loop][0] < threshold :
                        detected_activity.append(("picking",ind) )
                #elif(a1[ind][loop - 1][0] < threshold and a1[ind][loop][0] > a1[ind][loop-1][0]) and calculateDistance(wloc[loop][0] , wloc[loop-1][0]) >= movementThreshold:
                #    detected_activity.append(("placing",ind) )
        
            if a2[ind][loop][0] != -1 and a2[ind][loop - 1][0] != 1 :     
                if a2[ind][loop][0] <= a2[ind][loop-1][0] and a2[ind][loop][0] < threshold :
                        detected_activity.append(("picking",ind) )
                #elif(a2[ind][loop - 1][0] < threshold and a2[ind][loop][0] > a2[ind][loop-1][0]) and calculateDistance(wloc[loop][1] , wloc[loop-1][1]) >= movementThreshold:
                #    detected_activity.append(("placing",ind) )
        if w[loop][0] != -1 and w[loop][1] != -1:
            if w[loop][0] <= w[loop - 1][0] and w[loop][0] <= bagThreshold:
                detected_activity.append(("cart",8) )
            if w[loop][1] <= w[loop - 1][1] and w[loop][1] <= bagThreshold:
                detected_activity.append(("cart",8) )
    return detected_activity


def check_movement(obj , threshold):
    loop = 1
    dist_over_time = []
    dist_over_time.append(calculateCentroid(obj[0]))
    flag =False
    while loop < len(obj):
        dist_over_time.append(calculateCentroid(obj[loop]))
        if obj[loop].xtop !=0 and obj[loop].ytop !=0 and obj[loop-1].xtop !=0 and obj[loop].ytop !=0: 
            if calculateDistance(calculateCentroid(obj[loop]) , calculateCentroid(obj[loop - 1])) >= threshold:
                flag = True
        loop += 1
    return flag
            
def check_wrist_movement(obj , threshold):
    loop = 1
    flag = False
    while loop < len(obj):
        if obj[loop][0][0] !=-1:
            if calculateDistance(obj[loop][0] , obj[loop - 1][0]) >= threshold:
                flag = True
        if obj[loop][1][0] !=-1:
            if calculateDistance(obj[loop][1] , obj[loop - 1][1]) >= threshold:
                flag = True
        loop += 1
    return flag

def activityRecognitionThreshold(sequence,threshold , movementThreshold):
    #Input: look at sequence class
    #threshold: the minimum distance to be recognized.
    #ouput: activity
    detected_activity = []
    person_set = generate_person_object_locations_prevframes(sequence)
    for person in sorted(person_set.keys()):
        person_list = person_set[person]

        # iterate and see how the distances vary between objects and wrist locations for each person
        obj_over_time_lwrist = {}
        obj_over_time_rwrist = {}
        obj_over_time_bag = {}
        obj = {}
        for i in range(1,8):
            obj_over_time_lwrist[i] = []
            obj_over_time_rwrist[i] = []
            obj_over_time_bag[i] = []
            obj[i] = []
        wrist_over_time = []
        wrist_loc_over_time = []
        
        # iterate over each frame person list contains (object_locations, joint location) at frame iteration
        for item in person_list:
            bag, others = get_bag_and_other_objects(item[0])
            if len(bag) == 0:
                bag = Object()
                bag.type = 8
            else:
                bag = bag[0]

            per = item[1]
            if len(per.jointLocations) == 0:
                lwristloc = (-1, -1)
                rwristloc = (-1, -1)
                wrist_over_time.append((-1,-1))
                wrist_loc_over_time.append( ( lwristloc , rwristloc ) )
            else: 
                lwristloc = (per.jointLocations[14] , per.jointLocations[15] )
                rwristloc = (per.jointLocations[8] , per.jointLocations[9] )
                if calculateCentroid(bag) == (0,0):
                    wrist_over_time.append((-1,-1))
                else:
                    wrist_over_time.append( ( calculateDistance(lwristloc,calculateCentroid(bag)) , calculateDistance(rwristloc,calculateCentroid(bag)) ) )
                wrist_loc_over_time.append( ( lwristloc , rwristloc ) )

            for index in range(1,8):    #based on the number of objects
                flag = 0
                for o in others:
                    if index == o.type:
                        obj[index].append(o)
                        obj_over_time_lwrist[index].append( (calculateDistance(calculateCentroid(o) , lwristloc) , o))
                        obj_over_time_rwrist[index].append( (calculateDistance(calculateCentroid(o) , rwristloc) , o))
                        obj_over_time_bag[index].append(    (calculateDistance(calculateCentroid(o) , calculateCentroid(bag)) , o))
                        flag = 1
                if flag == 0:
                    o = Object()
                    o.type = index
                    obj_over_time_lwrist[index].append((-1,o))
                    obj_over_time_rwrist[index].append((-1,o))
                    obj_over_time_bag[index].append((-1,o))
                    #appending the distances as well as the object location at that instance

        #check if the object is being picked,added or dropped over the past five frames
        det_activity = check_activity(obj_over_time_lwrist,obj_over_time_rwrist,obj_over_time_bag,wrist_over_time , wrist_loc_over_time, threshold,movementThreshold)
        mod_activity = []
        for item in det_activity:
            act = item[0]
            num = item[1]
            if act != "cart":
                if check_movement(obj[num] , 10):
                    mod_activity.append(item )
            else:
                if check_wrist_movement(wrist_loc_over_time , 10):
                    mod_activity.append(item )
        detected_activity.append( (person, mod_activity) )
    
    return detected_activity

########################################################################################################################
########################################################################################################################

def generate_featuremap_lstm(sequence):
    #output: generates each feature over a time frame individually.
             #feature1 [x x1 x2 x3 x4]
             #feature2 [y y1 y2 y3 y4]
             # feature here means the locations of objects and joint locations.
    person_set = generate_person_object_locations_prevframes(sequence)
    feature_list = []
    
    for item in person_set.keys():
        name = item
        item = person_set[item]
        #usually 8 objects (1 + 7 objects ) and 18 joint locations == total
        features = [] # over frames
        for i in range(5):
            features.append([])
        idx = 0
        for frameitem in item:  #iterating over frames and appending features
            # appending object locations
            if len(frameitem[0]) == 0:
                for i in range(8):
                    features[idx].extend([-1, -1])
            else:
                for obj in frameitem[0]:
                    features[idx].extend(calculateCentroid(obj))
            
            joint_locations = frameitem[1].jointLocations
            if len(joint_locations) == 0:
                for i in range(8,26):
                    features[idx].extend([ -1, -1 ])
            else:
                j = 0
                for i in range(8,26):
                    features[idx].extend([ joint_locations[j],joint_locations[j+1] ])    
                    j += 2
            idx += 1
        feature_list.append((name,features))
    return feature_list
         
from numpy import array
def convert_to_lstm_format(all_features):
        npfeatures = array(all_features) 	
        features = npfeatures.reshape(len(all_features), 5, 52)
        print(features.shape)
        return npfeatures      

def normalize_time_series(all_features):
    new_list = []

    #normalization
    for item in all_features:
        for item2 in item:
            new_list.append(item2)
    
    from sklearn.preprocessing import MinMaxScaler  
    scaler = MinMaxScaler(feature_range = (0, 1))
    all_features_scaled = scaler.fit_transform(new_list)  
    
    final_list = []
    lis= []
    #converting back to time series
    for idx in range(len(all_features_scaled)):
        lis.append(all_features_scaled[idx])
        if (idx + 1) % 5 ==0:
            final_list.append(lis)
            lis=[]
    return final_list

def load_trained_model():
    model = load_model('77.h5')
    return model

def activity_detection(cursequence):
    features = generate_featuremap_lstm(cursequence)
    persons = []
    mod_features = []
    for item in features:
        mod_features.append(item[1])
        persons.append(item[0])
    features= mod_features
    features = normalize_time_series(features)
    mod_features = convert_to_lstm_format(features)
    model =load_trained_model()
    test_pred = model.predict(mod_features)
    detected_activity = []
    for item,idx in enumerate(test_pred):
        loc = -1
        for i in range(4):
            if item[i] == 1:
                loc = i
        if loc == 0:
            act = "picking"
        elif loc == 1:
            act = "placing"
        elif loc == 2:
            act = "cart"
        elif loc == 3:
            act = "idle"
        detected_activity.append((persons[idx] , act))
    return detected_activity

def parse_file(filename):
    #input:filepath to persondetection output
    #output: txt files with person details for each frame


    return 

def system_worker():
    # input: none
    # ouput: none
    # task : entire system
    
    
    #video_path = ""
    frame_path = "./frames/"
    object_detection_path = "./labels/"
    pose_estimation_path = "./persondata/"
    inventory_file_path = "./inventory.txt"
    video_path = ''
    #splitter_module(video_path, frame_path)
    FillStoreInventory(inventory_file_path)

    #object_detection(frame_path)
    #person_detection_file = person_detection(frame_path)
    #parse_file(person_detection_file)

    #make sure that the number of frames outputted is greater than 5.
    cur_frame_number = 57
    frame_array = []
    size  = (0,0)
    while(1):
        print(cur_frame_number)
        cursequence = Sequence()
        cursequence.dirName = cur_frame_number
        imgdatalist = []
        break_flag = 0
        for counter in range(0,5):
            curimage = Image()
            frame_index = cur_frame_number - 5 + counter
            #print(frame_index)
            object_filepath = object_detection_path + str(frame_index) + ".txt"
            person_filepath = pose_estimation_path + str(frame_index) + ".txt"
            objects_in_frames = []
            persons_in_frames = []
            try:
                obj_read = open(object_filepath, 'r')
                person_read = open(person_filepath, 'r')
            except:
                print("file not found", cur_frame_number)
                inp = int(input("do you want to continue 1 continue and 0 break?"))
                if inp == 0:
                    break_flag = 1
                    break
            ###READING OBJECT LOCATIONS
            objects = []
            for line in obj_read:
                curobject = Object()
                input_numbers = line.rstrip('\n').split(' ')
                if len(input_numbers) == 7:
                    curobject.type = int(input_numbers[0])
                    curobject.xtop = int(input_numbers[1])
                    curobject.ytop = int(input_numbers[2])
                    curobject.xbot = int(input_numbers[3])
                    curobject.ybot = int(input_numbers[4])
                    curobject.xres = int(input_numbers[5])
                    curobject.yres = int(input_numbers[6])
                    objects.append(curobject)
                else:
                    print("Read ERROR len not 7")
            curimage.objectList = objects
            objects_in_frames.append((cur_frame_number,objects))
    
            ###READING PERSON DETAILS
            persons = []
            for line in person_read:
                curperson= Person()
                input_numbers = line.split(' ')
                #print( input_numbers)
                if len(input_numbers) == 42:   #as it inlcudes '\n' at the end. usually its 41
                    curperson.name = input_numbers[0]
                    curperson.xtop = int(input_numbers[1])
                    curperson.ytop = int(input_numbers[2])
                    curperson.xbot = int(input_numbers[3])
                    curperson.ybot = int(input_numbers[4])
                    for idx in range(5,41):
                        curperson.jointLocations.append(int(input_numbers[idx]))
                else:
                    print("Read ERROR len not 41 ")
                persons.append(curperson)
                
            persons_in_frames.append((cur_frame_number,persons))
            curimage.personList = persons
            imgdatalist.append(curimage)
        
        if break_flag == 1:
                break
        cursequence.imageDataList = imgdatalist
        # use one of these.    
        # lstm based activity detection
        #detected_activity = activity_detection(cursequence)

        #threshold based activity detection
        detected_activity = activityRecognitionThreshold(cursequence, 25 , 25)
        print(detected_activity)

        picking =set()
        cart = set()
        for item in detected_activity:
            person_name = item[0]
            #for each person iterate through the detected activities.
            for it in item[1]:
                activity = it[0]
                obj_idx = it[1]
                is_there = 0
                if person_name in person_details.keys():
                    is_there = 1
                if activity == "picking":
                    if is_there:
                        pd = person_details[person_name]
                        it = pd.pickedUpItems
                    else:
                        it = set()
                        pd = PersonDetails()
                    it.add(int(obj_idx))
                    obj_det = object_to_detail_mapper(int(obj_idx))
                    picking.add(obj_det.name) 
                    pd.pickedUpItems = it
                    person_details[person_name] = pd
                elif activity == "placing":
                    #check if its in picked up items and drop it.
                    if is_there:
                        pd = person_details[person_name]
                        it = pd.pickedUpItems
                        if obj_idx in it:
                            it.remove(obj_idx)
                        pd.pickedUpItems = it
                        person_details[person_name] = pd
                elif activity == "cart":
                    # add all the picked up items into the cart.
                    if is_there:
                        pd = person_details[person_name]
                        it = pd.pickedUpItems
                        ct = pd.CartItems
                        for item in it:
                            obj_det = object_to_detail_mapper(int(item))
                            cart.add(obj_det.name)
                            ct.add(item)
                        pd.pickedUpItems = set()
                        #empty the pickedup items
                        pd.CartItems = ct
                        person_details[person_name] = pd
       
        
        ##try to overlay the cart data on the frame data.
        cur_frame_path = frame_path + str(cur_frame_number)+".jpg"
        cur_img = cv2.imread(cur_frame_path, cv2.IMREAD_COLOR)
        #print(cur_img.shape)
        height, width, depth = cur_img.shape
        size = ( width , height)
        
        text = "picking: "
        for item in picking:
            text += item + " "
        cv2.putText(cur_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        text = "placing: "
        cv2.putText(cur_img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        text = "cart: "
        for item in cart:
            text += item + " "
        cv2.putText(cur_img, text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        
            
            
#        for it in sorted(person_details.keys()):
#            details = person_details[it]
#            text += it + " "
#            for item in details.CartItems:
#                obj_details = object_to_detail_mapper(item)
#                text+= obj_details.name + " "
#            count += 1       
#            cv2.putText(cur_img, text, (10 * count, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        frame_array.append(cur_img)
        img_name = "./images/" + str(cur_frame_number) + ".jpg"
        cv2.imwrite(img_name,cur_img)
        cur_frame_number += 1

    ## calculating the prices
    text = ""
    for it in sorted(person_details.keys()):
        details = person_details[it]
        total_price = 0
        lis = details.CartItems
        for idx in lis:
            item = idx
            obj_details = object_to_detail_mapper(item)
            print(obj_details.name , obj_details.price)
            total_price += obj_details.price
        text += it + " " + str(total_price)
#    img = np.zeros((768,432,3), np.uint8)
#    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
   
#    for i in range(20):
#        img_name = "./images/" + str(cur_frame_number) + ".jpg"
#        frame_array.append(img)
#        cv2.imwrite(img_name,img)
#        cur_frame_number += 1

    ## writing to a video file
    out = cv2.VideoWriter('./activity_video_2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    

    return 


system_worker()
