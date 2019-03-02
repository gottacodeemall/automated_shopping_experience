import pickle
obj = ''
with open('persondata.pkl', 'rb') as f:
    obj = pickle.load(f)

#print type(obj)

for idx, item in enumerate(obj):
    imgname = item[0].split('.')[0]
    pername = item[1]
    bbox = item[2]
    jointloc = item[3]
    detailspath = "./persondata/" + imgname + ".txt"
    appstr = item[1] + " " + str(bbox[0][0]) + " " + str(bbox[0][1]) + " " + str(bbox[2][0]) + " " + str(bbox[2][1]) +" ";
    count = 0
    for it in jointloc:
        if count > it[1]:
            continue
        while count != it[1]:
            print it[1] , count
            appstr += "-1 -1 "
            count += 1
        appstr += str(it[0][0]) + " " + str(it[0][1]) + " "
        count +=1
    while count != 18:
        appstr += "-1 -1 "
        count += 1

    appstr += "\n"
    print appstr
    f = open(detailspath, "a")
    f.write(appstr) 
