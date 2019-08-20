##  Imports
###############################################################################
import scipy.io as spio
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

##  Loads necessary files
###############################################################################
print('Loading necessary files...')

training_filename = 'training_data.mat'      ## Loads training data into training_file
training_file = spio.loadmat(training_filename, squeeze_me=True)

submission_filename = 'submission_data.mat'  ## Loads submission data into submission_file
submission_file = spio.loadmat(submission_filename, squeeze_me=True)

print('Files loaded. Importing data...')

train_test_raw = training_file['d']
full_indexes = training_file['Index']
full_classes = training_file['Class']

submission_raw = submission_file['d']

combined_classes_indexes = list(zip(full_indexes, full_classes))    ## Combines into one list to maintain relationship between class and index
random.shuffle(combined_classes_indexes)                            ## Shuffles the combined list
full_indexes[:], full_classes[:] = zip(*combined_classes_indexes)   ## Writes shuffled list back into arrays

test_size = 0.2         ## Determines how much of the training data is used as testing data

train_indexes = full_indexes[:-int(test_size * len(full_indexes))]  ## Separates out training indexes
train_classes = full_classes[:-int(test_size * len(full_classes))]  ## Separates out training classes

test_indexes = full_indexes[-int(test_size * len(full_indexes)):]   ## Separates out testing indexes
test_classes = full_classes[-int(test_size * len(full_classes)):]   ## Separates out testing classes

print('Data loaded...')

###############################################################################
class data_point(object):
    ## Initialisation script
    def __init__(self, data, index, classification = 0):
        self.index = index  ## Sets the value of the data point index
        self.classification = classification ## Sets the classification of the data point, defaults to 0 if classification is unknown
        self.raw_data = []  ## Defines the array used to hold the raw data
        self.raw_data.append([0] * 64)  ## Allocates the array to contain 64 values
        self.raw_data = data[(index - 8):(index + 56)]  ## Imports raw data, from 8 before the index to the 56th value after the index

    ## Uses the raw data of this instance to determine a list of features
    def feature_analysis(self):
        fft_features = 6    ## Determines how many fft components are added to the end of the features list
        self.features = np.array([0.00] * (13 + fft_features))
        
        self.features[0]  = np.mean(self.raw_data)              ## Feature 0  - Data Mean
        self.features[1]  = np.std(self.raw_data)               ## Feature 1  - Data Standard Deviation
        self.features[2]  = np.amax(self.raw_data)              ## Feature 2  - Data Maximum Value
        self.features[3]  = np.amin(self.raw_data)              ## Feature 3  - Data Minimum Value
        self.features[4]  = np.median(self.raw_data)            ## Feature 4  - Data Median
        self.features[5]  = np.mean(self.raw_data[0:7])         ## Feature 5  - 1st 1/8 Mean
        self.features[6]  = np.mean(self.raw_data[8:15])        ## Feature 6  - 2nd 1/8 Mean
        self.features[7]  = np.mean(self.raw_data[16:23])       ## Feature 7  - 3rd 1/8 Mean
        self.features[8]  = np.mean(self.raw_data[24:31])       ## Feature 8  - 4th 1/8 Mean
        self.features[9]  = np.mean(self.raw_data[32:39])       ## Feature 9  - 5th 1/8 Mean
        self.features[10] = np.mean(self.raw_data[40:47])       ## Feature 10 - 6th 1/8 Mean
        self.features[11] = np.mean(self.raw_data[48:55])       ## Feature 11 - 7th 1/8 Mean
        self.features[12] = np.mean(self.raw_data[56:63])       ## Feature 12 - 8th 1/8 Mean
        
        ## Creates a list of features determined by the fft of the raw data
        ## Number of features generated determined by variable 'fft_features'
        fft_feature_list = self.determine_fft(fft_features)
        for i in range(13, (13 + fft_features)):
            self.features[i] = fft_feature_list[i-13]
        
    ## Initialises the components list for this instance to be populated later    
    def determine_components(self, num_components):
        self.components = np.array([0.00] * num_components) ## Initialises components array
        
    ## Determines the discrete fourier transform of the raw data of an instances
    def determine_fft(self, max_features):
        fft_full = np.fft.fft(self.raw_data)    ## Finds the fft of the raw data 
        fft_out = np.absolute(fft_full[0:max_features]) ## Determines the absolute value for the first 'max_features' fourier components
        
        return fft_out
    
###############################################################################
## Functions ##################################################################
###############################################################################
## Determines the indexes of any spikes within a dataset 'data'
##      Alpha defines the number of standard deviations greater than the mean boundary to classify a spike
##      Beta defines the number of standard deviations greater than the mean boundary dropped below to allow a new index to be identified again
        
def spike_identify_v1(data, alpha = 2.2, beta = 0.8):
    data_mean = np.mean(data)   ## Calculates global mean of data
    data_std = np.std(data)     ## Calculates global standard deviation of data
    
    indexes = 0
    index_list = []
    already_found = False
    
    data_length = len(data)
    ten_percent = int(data_length/10)  ## Value of 10% of 'data' length
    
    for i in range(0, data_length):
        if ((i % ten_percent) == 0):
            print('Percent complete: %.2f' % (100 * i/data_length),'%  Indexes Found - ', indexes)
        
        if data[i] > (data_mean + (alpha * data_std)): ## If value is greater than alpha boundary
            if already_found == False:  ## And was not above the boundary already
                already_found = True
                index_list.append([0] * (indexes + 1))  ## Appends the list length
                index_list[indexes] = i-2   ## Sets the index in the list
                indexes += 1
        else:
            if data[i] < (data_mean + (beta * data_std)):   ## If crosses the beta boundary
                already_found = False       ## Allows a new index to be found if alpha is crossed again 
    
    accepted_list = np.asarray(index_list)  ## Converts index list into numpy array
    
    return accepted_list    ## Returns list of generated indexes

###############################################################################
## Determines the indexes of any spikes within a dataset 'data'
##      'max_gradient' defines the gradient boundary, beyond which a spike can be detected
##      'min_gradient' defines the value at which the gradient must drop below before a new index can be generated
##      'gradient_dist' defines the number of data-points each side of the current used to determine the gradient       

def spike_identify_v2(data, max_gradient = 0.329, min_gradient = 0, gradient_dist = 3):
    indexes = 0
    index_list = []
    already_found = False
    
    
    data_length = len(data)
    ten_percent = int(data_length/10)  ## Value of 10% of 'data' length
    
    for i in range((0 + gradient_dist), (data_length - gradient_dist)):
        if ((i % ten_percent) == 0):
            print('Percent complete: %.2f' % (100 * i/data_length),'%  Indexes Found - ', indexes)
        
        if (find_gradient(data, i, gradient_dist) > max_gradient):
            if already_found == False:  ## And was not above the boundary already
                already_found = True
                index_list.append([0] * (indexes + 1))  ## Appends the list length
                index_list[indexes] = i-2   ## Sets the index in the list
                indexes += 1
        else:
            if (find_gradient(data, i, 3) < min_gradient):
                already_found = False       ## Allows a new index to be found if alpha is crossed again 
    
    accepted_list = np.asarray(index_list)  ## Converts index list into numpy array
    
    return accepted_list    ## Returns list of generated indexes

###############################################################################
## Determines if two indexes are within a range of '- min_range' to '+ max_range' from each other

def diff(indexA, indexB, min_range, max_range):
    matched = False
    for diff in range(-min_range, max_range+1):
        if (indexA == (indexB + diff)):
            matched = True
            break
        
    return matched

###############################################################################
## Finds the gradient in the data values around a point 'data_index', within range 'val_range'    

def find_gradient(data, data_index, val_range = 1):
    gradient = 0.00
    data_values = data[(data_index - val_range): ((data_index + 1) + val_range)]
    gradient = (data_values[(len(data_values) - 1)] - data_values[0])/(val_range + 1)
    return gradient

###############################################################################
## Applies PCA to both the training and testing data sets
    
def determine_PCA(training_data, testing_data, num_components = 10, scaling = True):
    scaler = StandardScaler() ## Defines standard scaler to be used
    print('PCA started...')
    training_features = np.zeros((len(training_data), len(training_data[0].features)))  ## Inits 2D array of features for each training data point
    testing_features = np.zeros((len(testing_data), len(testing_data[0].features)))  ## Inits 2D array of features for each testing data point
        
    for i in range(0, len(training_data)):      ## Imports features into 2D array
        training_features[i,:] = training_data[i].features
    
    for i in range(0, len(testing_data)):       ## Imports features into 2D array
        testing_features[i,:] = testing_data[i].features
    
    if (scaling):   ## If scaling boolean is enabled
        print('Scaling data...')
        scaler.fit(training_features) # Fit to the training set only
    
        training_features = scaler.transform(training_features) # Apply scaling transform
        testing_features = scaler.transform(testing_features) # Apply scaling transform

        print('Scaling complete...')
    
    print('Performing PCA Transform...')
    pca = PCA(num_components)   ## Sets the output number of components to be 'num_components'
    pca.fit(training_features)  ## Fits to training set only
    
    training_components = pca.transform(training_features)  ## Applies PCA transform to training data
    testing_components = pca.transform(testing_features)  ## Applies PCA transform to testing data
       
    
    for i in range(0, len(training_data)):  ## Outputs components into class instances
        training_data[i].components = training_components[i,:]
    
    for i in range(0, len(testing_data)):  ## Outputs components into class instances
        testing_data[i].components = testing_components[i,:]
    
    print('PCA complete...')
    
    return 0
    
###############################################################################
## Applies k-nearest neighbour on the training dataset, returning the most voted class for data point testing_data

def knn(training_data, testing_data, k = 4):
    distances = np.zeros(( len(training_data), 2))  ## Initialises the distances array
       
    for j in range(len(training_data)): ## For every data point in the training set
       distance = np.linalg.norm(np.array(training_data[j].components) - np.array(testing_data.components)) ## Determines the distance between the datapoint and the testing datapoint components
       distances[j] = ([distance, training_data[j].classification]) ## Adds calculated distance and datapoint classification
    
    sorted_distances = distances[distances[:,0].argsort()]  ## Sorts the distances 
    
    votes = [i[1] for i in sorted_distances[:k]]        ## Determines the classification of the k closest datapoints in the training set
    #print(Counter(votes).most_common(1))                ## Prints the most voted class and the number of votes for that class   
    vote_result = Counter(votes).most_common(1)[0][0]   ## Determines the most voted classification
    
    return vote_result  ## Returns predicted class

###############################################################################
## Applies smoothing to the dataset provided by applying an implementation of a Butterworth Bandpass Filter

def smooth_data(data, fs=25000, lowcut=10, highcut=1000, order=1):
    nyquist = 0.5*fs    ## Calculcates nyquist frequency of the data
    low = lowcut/nyquist    ## Calculates the lower critical frequency
    high = highcut/nyquist  ## Calculates the upper critical frequency
    
    b,a = butter(order, [low,high], btype='band')   ## Determines the numerator and denominator polynomials of the filter 
    data_smoothed = lfilter(b, a, data) ## Applies the butterworth filter to the data

    return data_smoothed

###############################################################################
## Outputs Indexes and Classes from 'data_points' instances into a '.mat' file
## Can use alternative filename by passing a string as an additional parameter

def output_to_mat(data_points, output_filename = '10097.mat'):
    class_data = np.zeros(len(data_points)) ## Inits class array
    class_data = class_data.astype("uint8") ## Sets to dtype 'uint8'
    index_data = np.zeros(len(data_points)) ## Inits index array
    index_data = index_data.astype("int32") ## Sets to dtype 'int32'
    
    for i in range(0,len(data_points)): ## For every data point
        class_data[i] = data_points[i].classification   ## Get Class
        index_data[i] = data_points[i].index            ## Get Index
    
    a={}    ## Init output array
    a['Index'] = index_data     ## Import index array
    a['Class'] = class_data     ## Import class array
    
    spio.savemat(output_filename,a) ## Save file
    
    return 0   

###############################################################################
##  Booleans Used #############################################################
###############################################################################
PCA_Scaling = True                      ## Enables/Disables Scaling before performing PCA
close_program = False                   ## If enabled the program will close at the end of the current program loop
selection_valid = False                 ## Determines whether the input by the user is valid

###############################################################################
##  Main Section ##############################################################
###############################################################################
while (close_program != True):
    print("\n")
    print("0:  Close Program")
    print("1:  Test spike identification on the training dataset")
    print("2:  Display 2D plot of PCA components generated")
    print("3:  Test KNN classification efficiency on the testing dataset")
    print("4:  Find optimal KNN value for 'k' in range 4 to 500 (around 30h runtime)")
    print("5:  Display submission raw data against training/testing raw data")
    print("6:  Display submission raw data against submission smoothed data")
    print("7:  Display submission smoothed data against training/testing raw data")
    print("8:  Calculate Indexes and Classes for submission dataset")
    print("9:  Enable/Disable PCA standard scaling. Currently: ", PCA_Scaling)
    
    while(selection_valid == False):
        selection = int(input("Please enter your selection (0-9):  "))
        if ((selection >= 0) and (selection <= 9)):
            selection_valid = True
        else:
            print("Invalid Argument")
            selection_valid = False
        
###############################################################################
    if (selection == 0):    
        close_program = True
###############################################################################
    if (selection == 1):
        sorted_indexes = np.asarray(sorted(full_indexes, reverse=False))
        spike_selection_valid = False
        
        while(spike_selection_valid == False):
            spike_selection = int(input("Would you like to test v(1) or v(2):  "))
            if ((spike_selection > 0) and (spike_selection <= 2)):
                spike_selection_valid = True
            else:
                print("Invalid Argument")
                spike_selection_valid = False
        
        print('Starting to identify indexes...')
        if(spike_selection == 1):
            start = time.time()
            my_indexes = spike_identify_v1(train_test_raw)
            end = time.time()
            print('\nIndex identify run-time: %.2f' % (end-start), 'seconds')
        elif(spike_selection == 2):
            start = time.time()
            my_indexes = spike_identify_v2(train_test_raw)
            end = time.time()
            print('\nIndex identify run-time: %.2f' % (end-start), 'seconds')
            
        i=k=0
        matches = 0
        false_positives = 0
        not_found = 0
        min_range = 8
        max_range = 8
    
        while (i < len(sorted_indexes)) and (i+k < len(my_indexes)):
            if (diff(sorted_indexes[i], my_indexes[i+k], min_range, max_range)):
                matches += 1
                i += 1
            else:
                if(sorted_indexes[i] > (my_indexes[i+k] - max_range)):
                    while(sorted_indexes[i] > (my_indexes[i+k] - max_range)):
                        false_positives += 1    ## False positive detected
                        k += 1                  ## Try next index in my list
                        if ((k+i) >= len(my_indexes)):   ## In case of finishing here
                            break
                else:
                    not_found += 1  ## One of the indexes has been missed
                    i += 1          ## Got to next index in list
                    k -= 1          ## Keep the position in my list of indexes the same
                        
        spike_detection_efficiency = (matches/(len(sorted_indexes))) * 100
        print('Number of matches found: ', matches)
        print('Number of indexes not found: ', not_found)
        print('Number of false positives: ', false_positives)
        print('Spike identification efficiency: %.4f' % spike_detection_efficiency, '% \n')        
        #print(not_found, ' + ', matches, ' = ', (matches+not_found), ' : ', len(sorted_indexes))            ## Both lines used to test if all of the data has been searched
        #print(false_positives, ' + ', matches, ' = ', (matches+false_positives), ' : ', len(my_indexes))    ## Both additions should equal the len of the index arrays
        
###############################################################################
    if (selection == 2):
        
        scaler = StandardScaler() ## Defines standard scaler
        num_components = 2 ## Determines how many PCA components are generated
    
        training_data = [data_point(train_test_raw, train_indexes[i], train_classes[i]) for i in range(len(train_indexes))]
        testing_data = [data_point(train_test_raw, test_indexes[i], test_classes[i]) for i in range(len(test_indexes))]
    
        for i in range(0, len(training_data)):    ## Applies feature analysis
            training_data[i].feature_analysis()
            training_data[i].determine_components(num_components)
    
        for i in range(0, len(testing_data)):    ## Applies feature analysis
            testing_data[i].feature_analysis()
            testing_data[i].determine_components(num_components)
    
        determine_PCA(training_data, testing_data, num_components, PCA_Scaling)
    
    ## After this is only used for drawing the graph
    ###########################################################################    
    
        colors = ['r', 'g', 'b', 'y']   ## Defines colors used
        handles = []    ## Defines handles according to colours representing a class
        r_patch = mpatches.Patch(color='r', label='Class 1')
        g_patch = mpatches.Patch(color='g', label='Class 2')
        b_patch = mpatches.Patch(color='b', label='Class 3')
        y_patch = mpatches.Patch(color='y', label='Class 4')

    
        data_x = ([0.00] * len(training_data))  ## 
        data_y = ([0.00] * len(training_data))  ## Initialise
        data_c = (['n'] * len(training_data))   ##
    
        for i in range(0, len(training_data)):  ## For every value in 'training_data'
            data_x[i] = training_data[i].components[0]  ## Copy first principal component
            data_y[i] = training_data[i].components[1]  ## Copy second principal component
            data_c[i] = colors[training_data[i].classification - 1] ## Works out color of scatter point by the class of said principal component
        
    
        fig = plt.figure(figsize = (8,8))       ## Sets up the graph
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 Component PCA', fontsize = 20)
        ax.legend(handles=[r_patch, g_patch, b_patch, y_patch])
    
        ax.scatter(data_x, data_y, c = data_c, s = 20)   
        ax.grid()
        plt.pause(0.001)
        input("Press [enter] to continue")
###############################################################################
    if (selection == 3):

        scaler = StandardScaler() ## Defines standard scaler
        num_components = 10 ## Determines how many PCA components are generated
    
        training_data = [data_point(train_test_raw, train_indexes[i], train_classes[i]) for i in range(len(train_indexes))]
        testing_data = [data_point(train_test_raw, test_indexes[i], test_classes[i]) for i in range(len(test_indexes))]
    
        for i in range(0, len(training_data)):    ## Applies feature analysis
            training_data[i].feature_analysis()
            training_data[i].determine_components(num_components)
    
        for i in range(0, len(testing_data)):    ## Applies feature analysis
            testing_data[i].feature_analysis()
            testing_data[i].determine_components(num_components)
    
        determine_PCA(training_data, testing_data, num_components, PCA_Scaling)

    ###############################################################################    

        correct = total = 0
        percentile = int(len(testing_data)/10)
        
        start = time.time()
        for data in range(len(testing_data)):   ## For every data point in the testing set
            if ((data % percentile) == 0):
                print('KNN: Testing %.2f' % (100 * data/len(testing_data)),'% complete   (', data, '/', len(testing_data), ')')
            result = knn(training_data, testing_data[data]) ## Predict class using k-nearest neighbour
            if (result == testing_data[data].classification):   ## If prediction was correct
                correct += 1
            total += 1 
        end = time.time()
        
        print('\nKNN: Testing Complete\n')
        print('KNN: Classification run-time: %.2f' % (end-start), 'seconds')
        print('Classification Accuracy: %.4f' % ((correct/total)*100), '%')
    
###############################################################################
    if (selection == 4):
        
        scaler = StandardScaler() ## Defines standard scaler
        num_components = 10 ## Determines how many PCA components are generated
    
        training_data = [data_point(train_test_raw, train_indexes[i], train_classes[i]) for i in range(len(train_indexes))]
        testing_data = [data_point(train_test_raw, test_indexes[i], test_classes[i]) for i in range(len(test_indexes))]
    
        for i in range(0, len(training_data)):    ## Applies feature analysis
            training_data[i].feature_analysis()
            training_data[i].determine_components(num_components)
    
        for i in range(0, len(testing_data)):    ## Applies feature analysis
            testing_data[i].feature_analysis()
            testing_data[i].determine_components(num_components)
    
        determine_PCA(training_data, testing_data, num_components, PCA_Scaling)

    ###############################################################################    
    
        accuracies = []
    
        fig = plt.figure(figsize = (8,8))   ## Sets up the graph
        ax1 = fig.add_subplot(1,1,1) 
        min_k = 4   ## Number of voting groups
        max_k = 500 ## 1/6 length of training set approximately
        
        for k in range(min_k,max_k):
            correct = total = 0
            print("KNN: pass ", (k - (min_k - 1)), " of ", (max_k - (min_k - 1)))
            for data in range(len(testing_data)):
                result = knn(training_data, testing_data[data], k)
                if (result == testing_data[data].classification):
                    correct += 1
                total += 1 
            accuracy = (correct/total)*100
            accuracies.append(accuracy)
    
        ax1.plot(np.array(range(min_k,max_k)), accuracies)
        plt.pause(0.001)
        input("Press [enter] to continue")
###############################################################################
    if (selection == 5):
        print("Loading...")
        fig = plt.figure(figsize = (8,8))
    
        ax1 = fig.add_subplot(2,1,1) 
        ax1.set_xlabel('Submission Raw Data', fontsize = 15)
        ax1.plot(submission_raw)
       
        ax2 = fig.add_subplot(2,1,2) 
        ax2.set_xlabel('Training/Testing Raw Data', fontsize = 15)
        ax2.plot(train_test_raw)
    
        plt.pause(0.001)
        input("Press [enter] to continue")
###############################################################################
    if (selection == 6):
        print("Loading...")
        fig = plt.figure(figsize = (8,8))
    
        ax1 = fig.add_subplot(2,1,1) 
        ax1.set_xlabel('Submission Raw Data', fontsize = 15)
        ax1.plot(submission_raw)
       
        ax2 = fig.add_subplot(2,1,2) 
        ax2.set_xlabel('Submission Smoothed Data', fontsize = 15)
        ax2.plot(smooth_data(submission_raw))
    
        plt.pause(0.001)
        input("Press [enter] to continue")
###############################################################################
    if (selection == 7):
        print("Loading...")
        fig = plt.figure(figsize = (8,8))
        
        ax1 = fig.add_subplot(2,1,1) 
        ax1.set_xlabel('Training/Testing Raw Data', fontsize = 15)
        ax1.plot(train_test_raw)
       
        ax2 = fig.add_subplot(2,1,2) 
        ax2.set_xlabel('Submission Smoothed Data', fontsize = 15)
        ax2.plot(smooth_data(submission_raw))
    
        plt.pause(0.001)
        input("Press [enter] to continue")
###############################################################################
    if (selection == 8):
        num_components = 10 ## Determines how many PCA components are generated
        submission_smoothed = smooth_data(submission_raw)
    
        spike_selection_valid = False
        
        while(spike_selection_valid == False):
            spike_selection = int(input("Would you like to use spike identification v(1) or v(2):  "))
            if ((spike_selection > 0) and (spike_selection <= 2)):
                spike_selection_valid = True
            else:
                print("Invalid Argument")
                spike_selection_valid = False
               
        if(spike_selection == 1):
            submission_indexes = spike_identify_v1(submission_smoothed)
        elif(spike_selection == 2):
            submission_indexes = spike_identify_v2(submission_smoothed)
    
        training_data = [data_point(train_test_raw, train_indexes[i], train_classes[i]) for i in range(len(train_indexes))]
        submission_data = [data_point(submission_smoothed, submission_indexes[i]) for i in range(len(submission_indexes))]
    
        for i in range(0, len(submission_data)):    ## Applies feature analysis
            submission_data[i].feature_analysis()
            submission_data[i].determine_components(num_components)
        
        for i in range(0, len(training_data)):    ## Applies feature analysis
            training_data[i].feature_analysis()
            training_data[i].determine_components(num_components)
    
        determine_PCA(training_data, submission_data, num_components, PCA_Scaling)

    ###############################################################################    
        print("\nKNN: Testing Starting \n")
        for data in range(len(submission_data)):   ## For every data point in the testing set
            print("KNN: Testing ", (data+1), " of ", len(submission_data))
            submission_data[data].classification = knn(training_data, submission_data[data]) ## Predict class using k-nearest neighbour
        print('\nKNN: Testing Complete\n')
    
        output_to_mat(submission_data)
        print("Output file generated")

###############################################################################
    if (selection == 9):    ## Flips the value of 'PCA_Scaling' boolean
        if (PCA_Scaling):
            PCA_Scaling = False
        else:
            PCA_Scaling = True
###############################################################################    
    selection_valid = False     ## Stops from infinitely printing main section
        