import numpy as np 
import pandas as pd
import scipy 
import pickle
from scipy.spatial import distance as scidist
import sys
sys.path.insert(1, "D:\\Work\\Research\\")
from src.evaluate import distance
from sklearn.cluster import KMeans
import time
from collections import defaultdict
import os
import os.path

class DSI:
    def __init__(self, dataset):
        file_reader = open(dataset, 'rb')
        self.dataset = pickle.load(file_reader)
        if(type(self.dataset) != pd.core.frame.DataFrame):
            self.dataset = pd.DataFrame(self.dataset)
        self.variety = {}
        self.feature_centroids = {}
        self.simmat = {}
        self.class_index = {}
        self.class_list = []
        self.gen_index()
    
    def gen_variety_and_feature_centroids(self):
        model = KMeans(n_clusters=1)
        self.class_list = self.dataset['cls'].unique()
        for name in self.class_list:
            if(name not in self.variety.keys()):
                X = np.array(self.dataset.loc[self.dataset['cls'] == name]['hist'])
                X = np.vstack(X)
                model.fit(X)
                self.variety[name] = model.inertia_
                self.feature_centroids[name] = model.cluster_centers_
    
    def gen_similarity_matrix(self):
        model = KMeans(n_clusters=1)
        for index, name in enumerate(self.class_list):
            class_similarity = np.empty((len(self.class_list)))
            for second_index, second_name in enumerate(self.class_list):
                class_similarity[second_index] = distance(self.feature_centroids[name], self.feature_centroids[second_name], d_type="cosine")
            self.simmat[name] = class_similarity
            
    def gen_index(self):
        self.gen_variety_and_feature_centroids()
        self.gen_similarity_matrix()

start_time = time.time()
test = DCSI("D:\\Work\\Research\\Features - Dogs\\resnet-120")
print(round(time.time() - start_time, 1))

samples = test.dataset

breeds = np.unique(np.array(samples['cls'])).tolist()

class_variety = {}

# ## Variety contribution ratio 

for index, breed in enumerate(breeds):
    train_set = samples.loc[samples['cls'] == breeds[index]]
    
    train_set = train_set.drop('img', axis=1)

    X_train = train_set['hist']

    X_train

    from sklearn.cluster import DBSCAN

    X = []
    for value in X_train:
        temp_array = list(value)
        X.append(temp_array)

    model = DBSCAN(eps=0.04, min_samples=1, metric='cosine').fit(X)
    
    class_variety[breed] = len(set(model.labels_)) / len(X)

source = model.labels_

totalScore = 0 
for breed in class_variety.keys():
    variety_contribution_ratio.loc[len(variety_contribution_ratio.index)] = [breed.split("-")[-1], class_variety[breed]] 
    print("Breed: " + breed.split("-")[-1] + " has the variety ratio of: " + str(class_variety[breed]))
    totalScore += class_variety[breed]
    
(totalScore / len(breeds))**2

## For finding duplicates and transferring them to validation or maybe not

source = list(source)

def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs)>1)

master_filepath = "D:\\Work\\Research\\Stanford Dogs\\output\\train\\"

for index, breed in enumerate(breeds):
    sec_filepath = master_filepath + breeds[index].split("\\")[-1]
    
    train_set = samples.loc[samples['cls'] == breeds[index]]
        
    image_list = list(train_set['img'])
    
    train_set = train_set.drop('img', axis=1)

    X_train = train_set['hist']

    X_train

    from sklearn.cluster import DBSCAN

    X = []
    for value in X_train:
        temp_array = list(value)
        X.append(temp_array)

    model = DBSCAN(eps=0.04, min_samples=1, metric='cosine').fit(X)
    
    source = model.labels_
    
    source = list(source)
    
    for dup in sorted(list_duplicates(source)):
        original, repeat = dup
        try:
            repeat.remove(original)
        except:
            continue
        for item in repeat:
            filename = samples.loc[samples['img'] == image_list[item]]['img']
            filename = np.array(filename)
            filename = str(filename[0])
            filename = filename.split("\\")[-1]
            if(os.path.isfile(sec_filepath + "\\" + filename)):
                os.remove(sec_filepath + "\\" + filename)
    
    class_variety[breed] = len(set(model.labels_)) / len(X)