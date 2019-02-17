#import sys
#sys.path.insert(0, '/path/to/application/app/folder')

#from activity_detection import


def splitter_module(video_path,output_location):
    # input: video file location , output directory of frames
    # ouput: none
    # task : split video into frames

    return 

#global variables
person_details = []
person_cart = []
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
        self.pickedUpItems = []
        self.Wallet = 0
        self.inBagItems = []

class ObjectLocation:
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
                print("ERROR in inventory")
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
    for item in bags:
        ans = calculateDistance(calculateCentroid(item), calculateCentroid(person))
        if ans < min_dist:
            min_dist = ans
            closest_bag = item
    others.append(closest_bag)
    return others

############################################################################################################################################
#THRESHOLD BASED ACTIVITY DETECTION

previousFrame={}
picked_up_items={} #dictionary of person and picked up object
cart=[] # cart is list of sets.


def generate_person_object_locations_prevframes(prevobjects, prevpersons):
    person_set = {}
    # person_set is a dictionary mapping person name to the list of tuples (object, person) over the past 5 frames.
    # person_set["karthik"] = [(objects,person) , (objects,person) ,(objects,person) , (objects,person), (objects,person)]

    for i in range(0, len(prevobjects)):
        objects = prevobjects[i][1]
        persons = prevpersons[i][1]
        for person in persons:
            objectsmod = get_closest_bag(objects,person)
            if person.name in person_set.keys():
                lis = person_set[person.name]
                lis.append((objectsmod,person))
                person_set[person.name] = lis
            else:
                lis = []
                lis.append((objectsmod,person))
                person_set[person.name] = lis
    #also filters with only the bag closest to the person added to the objects corresponding to that person.
    # now for each person you have a the set of object locations and pose locations over the past 5 frames.
    return person_set


def check_activity(a1,a2,b,threshold, movementThreshold, bagThreshold):
    # based on the assumption that the object is unique(no duplicates)
    # if the previous frame and the current frame object distance decreases then 
        # if the object starts moving around then picking
    #else
        # if the object is moving and is close to the bag then add to the cart
        # if the objet is moving away from hand placing
    detected_activity = []
    length = len(a1[0])
    for loop in range(2,length):
        for ind in range(1,8):
            if a1[ind][loop][0] <= a1[ind][loop-1][0]:
                if a1[ind][loop][0] != 0 and a1[ind][loop][0] < threshold and (calculateCentroid(a1[ind][loop][1]) - calculateCentroid(a1[ind][loop -1 ][1]) > movementThreshold ) :
                    detected_activity.append(("add",ind) )
            else:
                detected_activity.append(("placed",ind) )
            if a2[ind][loop][0] <= a2[ind][loop-1][0]:
                if a2[ind][loop][0] != 0 and a2[ind][loop][0] < threshold and (calculateCentroid(a2[ind][loop][1]) - calculateCentroid(a2[ind][loop -1 ][1]) > movementThreshold ) :
                    detected_activity.append(("add",ind) )
            else:
                detected_activity.append(("placed",ind) )

            if b[ind][loop][0] < b[ind][loop-1][0]:
                if b[ind][loop][0] != 0 and b[ind][loop][0] < bagThreshold and (calculateCentroid(b[ind][loop][1]) - calculateCentroid(b[ind][loop -1 ][1]) > movementThreshold ) :
                    detected_activity.append(("cart",ind) )
            else:
                detected_activity.append(("placed",ind) )
    return detected_activity


def activityRecognitionThreshold(self, prevobjects,prevpersons,threshold):
    #object list list of tuples(name,coord)
    #personlist list of tuples (details,coord)
    person_set = generate_person_object_locations_prevframes(prevobjects, prevpersons)
    for person in person_set.keys():
        person_list = person_set[person]

        # iterate and see how the distances vary between objects and wrist locations for each person
        obj_over_time_lwrist = {}
        obj_over_time_rwrist = {}
        obj_over_time_bag = {}
        for i in range(1,8):
            obj_over_time_lwrist[i] = []
            obj_over_time_rwrist[i] = []
            obj_over_time_bag[i] = []
        
        # iterate over each frame person list contains (object_locations, joint location) at frame iteration
        for item in person_list:
            bag, others = get_bag_and_other_objects(item[0])
            per = item[1]
            lwristloc = (per.jointLocations[x] , per.jointLocations[x] )
            rwristloc = (per.jointLocations[x] , per.jointLocations[x] )

            for index in range(1,8):    #based on the number of objects
                for o in others:
                    if index == o.type:
                        obj_over_time_lwrist[index].append( (calculateDistance(calculateCentroid(o) , lwristloc) , o))
                        obj_over_time_rwrist[index].append( (calculateDistance(calculateCentroid(o) , rwristloc) , o))
                        obj_over_time_bag[index].append(    (calculateDistance(calculateCentroid(o) , calculateCentroid(bag)) , o))
                    else:
                        o = ObjectLocation()
                        obj_over_time_lwrist[index].append((0,o))
                        obj_over_time_rwrist[index].append((0,o))
                        obj_over_time_bag[index].append((0,o))
                        #appending the distances as well as the object location at that instance

        #check if the object is being picked,added or dropped over the past five frames
        detected_activity = check_activity(obj_over_time_lwrist,obj_over_time_rwrist,obj_over_time_bag)
    return detected_activity
                


def generate_featuremap(prevobjects, prevpersons):
    #output: generates a feature map for each person in the fashion [(person.name, featuremap) , (person.name, featuremap) .. ]
            # feature are generated for the last 5 frames
    person_set = generate_person_object_locations_prevframes(prevobjects,prevobjects)
    feature_list = []
    
    for item in person_set.keys():
        item = person_set[item]
        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = []
        joint_locations = item[1].jointLocations
        shoulder_pos_l = [0,0] #joint_locations[x]
        wrist_pos_l = [0,0] #joint_locations[x]
        shoulder_pos_r = [0,0] #joint_locations[x]
        wrist_pos_r = [0,0] #joint_locations[x]

        #feature_1 shoulder wrist distance
        feature1.append(calculateDistance(wrist_pos_l, shoulder_pos_l))
        feature1.append(calculateDistance(wrist_pos_r, shoulder_pos_r))

        #feature_2 
        head_loc = [0,0] #joint_locations[x]

        count = 7 #maximum number of objects in a frame
        bags, others = get_bag_and_other_objects(item[0])
        for obj in others:
            count = count - 1
            feature2.append(calculateDistance(wrist_pos_l,calculateCentroid(obj)))
            feature2.append(calculateDistance(wrist_pos_r,calculateCentroid(obj)))
            feature3.append(calculateDistance(head_loc,calculateCentroid(obj)))
            feature4.append(calculateDistance(calculateCentroid(bags[0]),calculateCentroid(obj)))
        while count > 0:
            feature2.append(0)
            feature2.append(0)
            feature3.append(0)
            feature4.append(0)
            count = count - 1
        feature = feature1 + feature2 + feature3 + feature4
        feature_list.append((item,feature))     
    return feature_list


from joblib import load

def system_worker():
    # input: none
    # ouput: none
    # task : entire system
    video_path = ""
    frame_path = ""
    object_detection_path = ""
    pose_estimation_path = ""
    person_detection_path = ""
    inventory_file_path = ""

    splitter_module(video_path, frame_path)
    FillStoreInventory(inventory_file_path)


    cur_frame_number = 0
    object_filepath = object_detection_path + cur_frame_number + ".txt"
    person_filepath = pose_estimation_path + cur_frame_number + ".txt"
    objects_in_frames = []
    persons_in_frames = []
    while(1):
        try:
            obj_read = open(object_filepath, 'r')
            person_read = open(person_filepath, 'r')
        except:
            choice = int(input(print("Wait for files to be rendered: 0 or 1")))
            if not choice:
                exit()
        
        ###READING OBJECT LOCATIONS
        objects = []
        for line in obj_read:
            curobject = ObjectLocation()
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
        objects_in_frames.append((cur_frame_number,objects))

        ###READING PERSON DETAILS
        persons = []
        for line in person_read:
            splitt = line.rstrip('\n').split(' ')
            curperson = Person()
            if len(splitt) == 23:
                curperson.name = input_numbers[0]
                curperson.xtop = int(input_numbers[1])
                curperson.ytop = int(input_numbers[2])
                curperson.xbot = int(input_numbers[3])
                curperson.ybot = int(input_numbers[4])
                for i in range(5,23):
                    curperson.jointLocations.append(int(input_numbers[i]))
                persons.append(curperson)
            else:
                print("Read ERROR len not 23")
        persons_in_frames.append((cur_frame_number,persons))

        if cur_frame_number >= 5:
            clf = load('activity_recognition.joblib')
            prev_objects = objects_in_frames[-5:0]
            prev_persons = persons_in_frames[-5:0]
            feature_list = generate_featuremap(prev_objects,prev_persons)
            #apply SVM

        cur_frame_number += 1


    return 
