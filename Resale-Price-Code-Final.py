#!/usr/bin/env python
# coding: utf-8

# In[2]:


# We will import the relevant files
import csv
import numpy as np

# read_csv function
def read_csv(csvfilename):
    rows = []
    with open(csvfilename) as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            rows.append(row)
    # convert the list into Numpy array
    rows = np.array(rows)
    return rows

firstData = read_csv("resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv")
secondData = read_csv("resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv")


# In[3]:


firstData


# In[4]:


secondData


# In[5]:


# We will now join the csv together from 2015 to 2021
data = np.concatenate((firstData,secondData[1:]))
data


# In[6]:


# 'MULTI-GENERATION' and 'EXECUTIVE' do not have a fixed number of rooms.
# We will exclude these entries from the data.

def removeUnlabelledData(data):
    data1 = []
    for i in range(1,len(data)):
        row = data[i]
        # row with executive and multi-generation
        if row[2] == 'EXECUTIVE' or row[2] == 'MULTI-GENERATION':
            continue;
        else:
            data1.append(row)
    
    # convert the list to numpy array
    data1 = np.array(data1)
    return data1

data1 = removeUnlabelledData(data)
data1


# In[7]:


# We will retrieve the MRT Station from the MRT CSV file
mrtData = read_csv("MRTstations.csv")
mrtData


# In[8]:


# library needed for the operation to happen
import json
import requests

# We will create a function to return dictionary of MRT stations with their longitudes and lattitudes
def createStationLocation(data):
    # create dictionary
    mrtStationCoordinates = {}
    # iterate through the stations in mrtData
    for i in range(1,len(data)):
        row = data[i]
        # LRT Station
        if (row[2] == 'SK' or row[2] == 'Pu' or row[2] =='BP'):
            stationName = row[1] + ' LRT STATION'
        # MRT Station
        else:
            stationName = row[1] + ' Mrt station'
        # print(stationName)
        # we will use the Station code to query the longitude and lattitude from the onemap api provided by SLA
        link = "https://developers.onemap.sg/commonapi/search?searchVal=%s&returnGeom=Y&getAddrDetails=Y" % stationName
        # print(link)
        resp = requests.get(link)
        
        # print(json.loads(resp.content))
        # we will narrow down to the the result section from the dictionary
        query = json.loads(resp.content)['results'][0]
        # print(query)
        
        # initialise the nested dictionary
        mrtStationCoordinates[row[1]] = {}
        
        # add the lattitude and convert to float value
        mrtStationCoordinates[row[1]]['Latitude'] = float(query['LATITUDE'])
        
        # add the longitude and convert to float value
        mrtStationCoordinates[row[1]]['Longtitude'] = float(query['LONGTITUDE'])
    
    return mrtStationCoordinates
        
mrtStationCoordinates = createStationLocation(mrtData)
mrtStationCoordinates


# In[9]:


# function to create dictionary of street name mapping to the their nearest mrt station
def createStreetNameLocation(data,mrtStationCoordinates):
    # intialise the dictionary
    streetNameLocation = {}
    
    # We will iterate the data
    for i in range(1,len(data)):
        row = data[i]
        streetName = row[4]
        # check if the street name is in the streetNameLocation
        if row[4] in streetNameLocation:
            # move on to the next row
            continue
        else:
            # ST George road
            if row[4] == "ST. GEORGE'S RD":
                # We will use this bus code instead
                postalCode = 60241
                link = "https://developers.onemap.sg/commonapi/search?searchVal=%s&returnGeom=Y&getAddrDetails=Y" % postalCode
            else:
                # We use onemap api to find the streetname coordinates
                link = "https://developers.onemap.sg/commonapi/search?searchVal=%s&returnGeom=Y&getAddrDetails=Y" % streetName            
            
            resp = requests.get(link)
                        
            # we will narrow down to the the result section from the dictionary
            query = json.loads(resp.content)['results'][0]
            
            # retrieve the list from this nearest MRT Station
            information = findNearestMRTStation(float(query['LATITUDE']),float(query['LONGTITUDE']),mrtStationCoordinates)
            
            # create a nested dictionary
            streetNameLocation[row[4]] = {}
            
            # add the nearest MRT station into the dictionary -- this is for debugging if any
            streetNameLocation[row[4]]["Nearest MRT"] = information[0]
            
            # add the nearest distance into the dictionary -- this will be used later
            streetNameLocation[row[4]]["Nearest Distance To MRT/LRT"] = information[1]            
            
            # add the nearest distance to CDB into the dictionary -- this will be used later
            streetNameLocation[row[4]]["Nearest Distance to CBD"] = information[2]
    
    return streetNameLocation           
        
    
# function that return a list that contains the name of nearest Mrt Station, 
# distance to nearest mrt station and the distance to cbd.
def findNearestMRTStation(latitude,longitude,mrtStationCoordinates):
    nearestMRT = ''
    nearestDistanceToMRT = 10000000
    nearestDistanceToCBD = 0
    information = [] 
    
    temp = 0   
    # we will iterate through the mrtStationCoordinates
    for key in mrtStationCoordinates:
        stationLatitude = mrtStationCoordinates[key]['Latitude']
        stationLongitude = mrtStationCoordinates[key]['Longtitude']
        temp = distanceCalculator(latitude,longitude,stationLatitude,stationLongitude)
        if temp < nearestDistanceToMRT:
            # reupdate the nearestDistanceToMRT
            nearestDistanceToMRT = temp
            # reupdate the stationame
            nearestMRT = key 
          
    # find the nearestDistanceToCBD
    nearestDistanceToCBD = distanceCalculator(latitude,longitude,
                                              mrtStationCoordinates['Raffles Place']['Latitude'],
                                              mrtStationCoordinates['Raffles Place']['Longtitude'])
    
    information.append(nearestMRT)
    information.append(nearestDistanceToMRT)
    information.append(nearestDistanceToCBD)
    
    # return the list
    return information    

import math
from math import radians,cos,sin,asin
# function that return the distance between 2 coordinates
def distanceCalculator(latitude1, longitude1,latitude2, longitude2):
    deltaLon = (longitude2- longitude1) * (math.pi/180)
    deltaLat = (latitude2 - latitude1) * (math.pi/180)
    a = (math.sin(deltaLat/2))**2 + cos(latitude1) * cos(latitude2)* ((sin(deltaLon/2))**2)
    
    c = 2 * math.asin(min(1,math.sqrt(a)))
    
    earthRadius = 6371 * 1000
    
    # find the distance 
    distance = earthRadius * c
    
    return distance


streetNameLocation = createStreetNameLocation(data1,mrtStationCoordinates)
streetNameLocation


# In[10]:


# We will now create getter functions to return the remaining data
def retrieveYear(entry):
    year = float(entry[:4])
    months = float(entry[5:7])/13
    time = year + months
    return time

def retrieveRoom(entry):
    return float(entry[0])

def retrieveMeanStorey(entry):
    # We will focus on the first 2 digits of the entry
    firstPart = entry[0]
    secondPart = entry[1]
    
    # single digit storey
    if firstPart == '0':
        # convert second digit to float
        startStorey = float(secondPart)
        mean = (startStorey + startStorey + 2)/2
        return mean        
    else:
        # convert first and second digit to float
        startStorey = float(entry[:2])
        mean = (startStorey + startStorey + 2)/2
        return mean

def retrieveFloorArea(entry):
    return float(entry)

def retrieveRemainingLease(entry):
    # only digits
    if 'years' in entry:
        if 'months' in entry:
            years = float(entry[:2])
            months = entry[9:12]
            # bad entry '0 m' representation in 2017 to 2021 csv
            if months == '0 m':
                return years
            else:
                months = float(entry[9:12])/12          
                time = years + months
                return time
        else:
            years = float(entry[:2])
            return years
    else:
        return float(entry)
    
def retrieveDistanceToMrt(entry,streetNameLocation):
    return streetNameLocation[entry]["Nearest Distance To MRT/LRT"]

def retrieveDistanceToCbd(entry,streetNameLocation):
    return streetNameLocation[entry]['Nearest Distance to CBD']

def retrievePrice(entry):
    return float(entry)


# In[11]:


# We will now initialise the numpy with zeros 
# https://www.dataquest.io/blog/numpy-tutorial-python/
xBlinded = np.zeros((len(data1),7))
xBlinded


# In[12]:


y = np.zeros((len(data1),1))
y


# In[13]:


# We will repopulate the data with independent and dependent variables
def populateBlindedX(data,xBlinded,streetNameLocation):
    for i in range(1,len(data)):
        row = data[i]
        # add the year
        xBlinded[i-1][0] = retrieveYear(row[0])
        
        # add the room number
        xBlinded[i-1][1] = retrieveRoom(row[2])
        
        # add the storey
        xBlinded[i-1][2] = retrieveMeanStorey(row[5])
        
        # add the floor area
        xBlinded[i-1][3] = retrieveFloorArea(row[6])
        
        # add the remaining lease
        xBlinded[i-1][4] = retrieveRemainingLease(row[9])
        
        # add the Distance to MRT/LRT
        xBlinded[i-1][5] = retrieveDistanceToMrt(row[4],streetNameLocation)
        
        # add the distance to CBD
        xBlinded[i-1][6] = retrieveDistanceToCbd(row[4],streetNameLocation)
              
    return xBlinded

xBlinded = populateBlindedX(data1,xBlinded,streetNameLocation)
np.set_printoptions(suppress=True,precision = 2)
xBlinded


# In[14]:


# Create the unblinded dataset
# We will now initialise the numpy with zeros
xUnblinded = np.zeros((len(data1),33))
xUnblinded


# In[15]:


# Return dictionary of town names that map to the location in the array
def retrieveTownNames(data1):
    townName = {}
    count = 7
    for i in range(1,len(data1)):
        row = data1[i]
        if row[1] not in townName:
            townName[row[1]] = count
            # inclement count + 1
            count += 1
        else:
            continue
    return townName
townName = retrieveTownNames(data1)
townName


# In[17]:


# We will create our unblinded dataset

def populateUnblindedX(data,xUnblinded,streetNameLocation,townName):
    for i in range(1,len(data)):
        row = data[i]
        # add the year
        xUnblinded[i-1][0] = retrieveYear(row[0])
        
        # add the room number
        xUnblinded[i-1][1] = retrieveRoom(row[2])
        
        # add the storey
        xUnblinded[i-1][2] = retrieveMeanStorey(row[5])
        
        # add the floor area
        xUnblinded[i-1][3] = retrieveFloorArea(row[6])
        
        # add the remaining lease
        xUnblinded[i-1][4] = retrieveRemainingLease(row[9])
        
        # add the Distance to MRT/LRT
        xUnblinded[i-1][5] = retrieveDistanceToMrt(row[4],streetNameLocation)
        
        # add the distance to CBD
        xUnblinded[i-1][6] = retrieveDistanceToCbd(row[4],streetNameLocation)
        
        # add the 1 to the correct location
        xUnblinded[i-1][townName[row[1]]] = 1
              
    return xUnblinded

xUnblinded = populateUnblindedX(data1,xUnblinded,streetNameLocation,townName)
np.set_printoptions(suppress=True)
xUnblinded


# In[19]:


def populateY(data,y):
    for i in range(1,len(data)):
        row = data[i]
        y[i-1][0] = float(row[10])
    return y

y = populateY(data1,y)
y


# In[20]:


# Feature scaling to ensure all the variables are equally represented
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[21]:


# Feature Scaling for the xBlinded dataset
xBlindedFS = sc.fit_transform(xBlinded)
xBlindedFS


# In[22]:


import statsmodels.api as sm

# Add the constant variable once
xBlindedFS1 = sm.add_constant(xBlindedFS)
xBlindedFS1


# In[23]:


# We assumed the name of town does not influence the price of the HDB.
model = sm.OLS(y,xBlindedFS1).fit()

# Run the predictions
predictions1 = model.predict(xBlindedFS1)

# Add the variables name 
#(https://stackoverflow.com/questions/36561897/naming-explanatory-variables-in-regression-output)
print(model.summary2(xname=['Constant','Year','Room Number','Storey','floor area',
                            'Remaining Lease','Distance To MRT/LRT',
                            'Distance To CBD'],yname = 'resale_price'))


# In[25]:


# find the mean of Resale Price
mean = np.mean(y)
mean


# In[27]:


# Regression error for xBlinded dataset
from sklearn import metrics

MAE1 = metrics.mean_absolute_error(y,predictions1)
MSE1 = metrics.mean_squared_error(y,predictions1)
RMSE1 = np.sqrt(metrics.mean_squared_error(y,predictions1))

print('Mean Absolute Error:', MAE1)
print('Mean Squared Error:', MSE1)
print('Root Mean Squared Error:', RMSE1)


# In[28]:


# Error rate

ErrorRate1 = (RMSE1/mean)*100
print(ErrorRate1)


# In[29]:


# Feature scaling for the xUnblinded
# Feature Scaling for the overall X dataset
xUnblindedFS = sc.fit_transform(xUnblinded)
xUnblindedFS


# In[30]:


# We will create our constant to the X column
xUnblindedFS1 = sm.add_constant(xUnblindedFS)
xUnblindedFS1


# In[31]:


# We assumed the name of town does not influence the price of the HDB.
model = sm.OLS(y,xUnblindedFS1).fit()

# Run the predictions
predictions2 = model.predict(xUnblindedFS1)

# Add the variables name 
print(model.summary2(xname = ['Constant',
                             'Year',
                             'Room Number',
                             'Storey',
                             'floor area',
                             'Remaining Lease',
                             'Distance To MRT/LRT',
                             'Distance To CBD',
                             'Ang Mo Kio',
                             'Bedok',
                             'Bishan',
                             'Bukit Batok',
                             'Bukit Merah',
                             'Bukit Panjang',
                             'Bukit Timah',
                             'Central Area',
                             'Choa Chu Kang',
                             'Clementi',
                             'Geylang',
                             'Hougang',
                             'Jurong East',
                             'Jurong West',
                             'Kallang/Whampoa',
                             'Marine Parade',
                             'Pasir Ris',
                             'Punggol',
                             'QueenTown',
                             'Sembawang',
                             'Sengkang',
                             'Serangoon',
                             'Tampines',
                             'Toa Payoh',
                             'Woodlands',
                             'Yishun'],
                             yname = 'resale_price'))


# In[32]:


# Regression error for X modified dataset
from sklearn import metrics

MAE2 = metrics.mean_absolute_error(y,predictions2)
MSE2 = metrics.mean_squared_error(y,predictions2)
RMSE2 = np.sqrt(metrics.mean_squared_error(y,predictions2))

print('Mean Absolute Error:', MAE2)
print('Mean Squared Error:', MSE2)
print('Root Mean Squared Error:', RMSE2)


# In[33]:


# Error rate

mean = np.mean(y)

ErrorRate2 = (RMSE2/mean)*100
print(ErrorRate2)


# In[ ]:




