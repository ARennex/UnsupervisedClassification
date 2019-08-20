# coding: utf-8
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Input, Concatenate, Conv1D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.models import model_from_json

import tensorflow as tf #Getting tf not defined errors

from sklearn.model_selection import StratifiedKFold, train_test_split
#from tqdm import tqdm

import numpy as np
import pandas as pd
import glob, os, random
import argparse

from collections import Counter #Count number of objects in each class

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gpus", type=int, default=1,
    help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
num_gpu = args["gpus"]

## Settings
# Files Setting
limit = 8000 # Maximum amount of Star Per Class Per Survey
extraRandom = True
permutation = True # Permute Files
BALANCE_DB = True # Balance or not
maximum = 5

# Mini Settings
MAX_NUMBER_OF_POINTS = 500
NUMBER_OF_POINTS = 500
n_splits = 10
validation_set = 0.2

# Iterations
step = 250

# Network Settings
verbose = True
batch_size = 2056
dropout = 0.5
hidden_dims = 128
epochs = 100

# Convolutions
filters = 128
filters2 = 64
kernel_size = 50
kernel_size2 = 50

base_path = os.getcwd()

#Laptop version
regular_exp1 = base_path + '/Temp/OGLE/**/*.dat'
regular_exp2 = base_path + '/Temp/ATLAS/**/*.csv'
regular_exp3 = base_path + '/Temp/VVV/**/*.csv'

subclasses = ['lpv','cep','rrlyr','ecl']

def get_files(extraRandom = False, permutation=False):
    files1 = np.array(list(glob.iglob(regular_exp1, recursive=True)))
    files2 = np.array(list(glob.iglob(regular_exp2, recursive=True)))
    files3 = np.array(list(glob.iglob(regular_exp3, recursive=True)))
    #Glob searches for all files that fit the format given in regular_exp1
    #Then puts them in a list

    print('[!] Files in Memory')

    # Permutations
    if permutation:
        files1 = files1[np.random.permutation(len(files1))]
        files2 = files2[np.random.permutation(len(files2))]
        files3 = files3[np.random.permutation(len(files3))]

        print('[!] Permutation applied')
        #Shuffles the arrays

    #Count the number of ogle cepheids in files1:
    ogle_count = {}
    for subclass in subclasses:
        ogle_count[subclass] = 0
    for ogle_file in files1:
        if 'cep' in ogle_file:
            ogle_count['cep'] += 1
        elif 'rrlyr' in ogle_file:
            ogle_count['rrlyr'] += 1
        elif 'lpv' in ogle_file:
            ogle_count['lpv'] += 1
        elif 'ecl' in ogle_file:
            ogle_count['ecl'] += 1
        else:
            print("this shouldn't happend")
            exit()
    print(ogle_count)

    aux_dic = {}
    ogle = {}
    ATLAS = {}
    vvv = {}
    for subclass in subclasses:
        aux_dic[subclass] = []
        ogle[subclass] = 0
        ATLAS[subclass] = 0
        vvv[subclass] = 0


    new_files = []
    #for idx in tqdm(range(len(files2))): #tqdm is a progress bar
    for idx in range(len(files1)): #tqdm is a progress bar
        foundOgle = False
        foundATLAS = False
        foundVista = False

        for subclass in subclasses:

            #Believe Cepheids may not being counted properly
            # if subclass == 'cep':
            #     print('Cepheid Test Print-out:')
            #     print(ogle[subclass])
            #     print(files1[idx])

            # Ogle
            # Limit is max stars of one class taken from survey (default 8000)
            if not foundOgle and ogle[subclass] < limit and subclass in files1[idx]:
                new_files += [[files1[idx], 0]]
                ogle[subclass] += 1
                foundOgle = True

            # ATLAS
            if not foundATLAS and ATLAS[subclass] < limit and idx < len(files2) and subclass in files2[idx]:
                new_files += [[files2[idx], 0]]
                ATLAS[subclass] += 1
                foundATLAS = True

            # VVV
            # idx check since VVV has less data than Ogle
            if not foundVista and vvv[subclass] < limit and idx < len(files3) and subclass in files3[idx]:
               new_files += [[files3[idx], 0]]
               vvv[subclass] += 1
               foundVista = True

    del files1, files2, files3
    #del files1, files2

    print('[!] Loaded Files')

    return new_files

def get_survey(path):
    if 'VVV' in path:
        return 'VVV'
    elif 'ATLAS' in path:
        return 'ATLAS'
    elif 'OGLE' in path:
        return 'OGLE'
    else:
        return 'err'

def get_name(path):
    for subclass in subclasses:
        if subclass in path:
            return subclass
    return 'err'

def get_name_with_survey(path):
    for subclass in subclasses:
        if subclass in path:
            survey = get_survey(path)
            return survey + '_' + subclass
    return 'err'

def open_vista(path, num):
    df = pd.read_csv(path, comment='#', sep=',', header = None)
    #print(df.iloc[0])
    df.columns = ['sourceID','mjd','mag','ppErrBits','Flag','empty']
    df = df[df.mjd > 0]
    df = df.sort_values(by=[df.columns[1]])

    # Something related to 3 standard deviations
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[1]].values, dtype=float)
    magnitude = np.array(df[df.columns[2]].values, dtype=float)
    error = np.array(df[df.columns[3]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time.astype('float'), magnitude.astype('float'), error.astype('float')

def open_ogle(path, num, n, columns):
    df = pd.read_csv(path, comment='#', sep='\s+', header=None)
    df.columns = ['a','b','c']
    df = df[df.a > 0]
    df = df.sort_values(by=[df.columns[columns[0]]])

    # Erase duplicates if it exist
    df.drop_duplicates(subset='a', keep='first')

    # 3 Desviaciones Standard
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]


    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time, magnitude, error

def open_atlas(path, num, n, columns):
    df = pd.read_csv(path, comment='#', sep=',', header=None)
    df.columns = ['a','b','c','d','e','f']
    try:
        df.b = df.b.astype(float)
    except Exception as e:
        print('Crashed while converting files! Crashing line was: ', df)
        print('File Location: ', path)
        exit()

    df = df[df.b > 0]
    df = df.sort_values(by=[df.columns[columns[0]]])

    # Erase duplicates if it exist
    df.drop_duplicates(subset='b', keep='first')

    # 3 Desviaciones Standard
    #df = df[np.abs(df.mjd-df.mjd.mean())<=(3*df.mjd.std())]

    time = np.array(df[df.columns[columns[0]]].values, dtype=float)
    magnitude = np.array(df[df.columns[columns[1]]].values, dtype=float)
    error = np.array(df[df.columns[columns[2]]].values, dtype=float)

    # Not Nan
    not_nan = np.where(~np.logical_or(np.isnan(time), np.isnan(magnitude)))[0]
    time = time[not_nan]
    magnitude = magnitude[not_nan]
    error = error[not_nan]

    # Num
    step = random.randint(1, 2)
    count = random.randint(0, num)

    time = time[::step]
    magnitude = magnitude[::step]
    error = error[::step]

    time = time[count:]
    magnitude = magnitude[count:]
    error = error[count:]


    if len(time) > n:
        time = time[:n]
        magnitude = magnitude[:n]
        error = error[:n]

    # Get Name of Class
    # folder_path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    # path, folder_name = os.path.split(folder_path)

    return time, magnitude, error

# Data has the form (Points,(Delta Time, Mag, Error)) 1D
def create_matrix(data, N):
    try:
        aux = np.append([0], np.diff(data).flatten())
    except Exception as e:
        print('Crashed at diff!')
        print('Value in Data List: ', data)
        exit()

    # Padding with zero if aux is not long enough
    if max(N-len(aux),0) > 0:
        aux = np.append(aux, [0]*(N-len(aux)))

    return np.array(aux[:N], dtype='float').reshape(-1,1)

def dataset(files, N):
    input_1 = []
    input_2 = []
    yClassTrain = []
    survey = []
    #for file, num in tqdm(files):
    for file, num in files:
        num = int(num)
        t, m, e, c, s = None, None, None, get_name(file), get_survey(file)
        if c in subclasses:
            if 'VVV' in file:
                t, m, e = open_vista(file, num)
            elif 'OGLE' in file:
                t, m, e = open_ogle(file, num, N, [0,1,2])
            elif 'ATLAS' in file:
                t, m, e = open_atlas(file, num, N, [1,3,4]) #These are the relevant columns in atlas data
            if c in subclasses:
                input_1.append(create_matrix(t, N))
                input_2.append(create_matrix(m, N))
                yClassTrain.append(c)
                survey.append(s)
            else:
                print('\t [!] E2 File not passed: ', file, '\n\t\t - Class: ',  c)
        else:
            print('\t [!] E1 File not passed: ', file, '\n\t\t - Class: ',  c)
    return np.array(input_1), np.array(input_2), np.array(yClassTrain), np.array(survey)


## Keras Model
def get_model(N, classes, activation='relu'):
    conv1 = Conv1D(filters, kernel_size, activation='relu')
    conv2 = Conv1D(filters2, kernel_size2, activation='relu')

    # For Time Tower
    input1 = Input((N, 1))
    out1 = conv1(input1)
    out1 = conv2(out1)

    # For Magnitude Tower
    input2 = Input((N, 1))
    out2 = conv1(input2)
    out2 = conv2(out2)

    out = Concatenate()([out1, out2])
    out = Flatten()(out)
    out = Dropout(dropout)(out)
    out = Dense(hidden_dims, activation=activation)(out)
    out = Dropout(dropout)(out)
    out = Dense(len(classes), activation='softmax')(out)

    model = Model([input1, input2], out)

    return model

def class_to_vector(Y, classes):
    new_y = []
    for y in Y:
        aux = []
        for val in classes:
            if val == y:
                aux.append(1)
            else:
                aux.append(0)
        new_y.append(aux)
    return np.array(new_y)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def experiment(files, Y, classes, N, n_splits):
    # Iterating
    activations = ['tanh']
    earlyStopping = [False]

    output = ''

    #Iterate over the activation functions, but only tanh is used where
    #Since it obtained the best results
    for early in earlyStopping:
        for activation in activations:
            # try:
            print('\t\t [+] Training',
                  '\n\t\t\t [!] Early Stopping', early,
                  '\n\t\t\t [!] Activation', activation)

            yPred = np.array([])
            yReal = np.array([])
            sReal = np.array([])

            modelNum = 0
            skf = StratifiedKFold(n_splits=n_splits)

            print('files',files)
            print('Y',Y)
            print('Y counts', Counter(Y))

            for train_index, test_index in skf.split(files, Y):
                print("Did split correctly?")
                dTrain, dTest = files[train_index], files[test_index]
                yTrain = Y[train_index]

                print("Did seperate into test + train correctly?")
                # Get Database
                dTrain_1, dTrain_2, yTrain, _ = dataset(dTrain, N)
                dTest_1, dTest_2, yTest, sTest  = dataset(dTest, N)

                print("Did run dataset function correctly?")

                yReal = np.append(yReal, yTest)
                sReal = np.append(sReal, sTest)
                yTrain = class_to_vector(yTrain, classes)
                yTest = class_to_vector(yTest, classes)

                print("Did vectorize class correctly?")

                #model = tf.keras.models.model_from_json(json_string)

                # load json and create model
                json_file = open(base_path + '/Results10Fold/tanh/model/9.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                loaded_model.load_weights(base_path + "/Results10Fold/tanh/model/9.h5")
                print("Loaded model from disk")

                from keras import backend as K
                frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in loaded_model.outputs])

                #loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                yPred = np.append(yPred, np.argmax(loaded_model.predict([dTest_1, dTest_2]), axis=1))

                #print(dTest_1,dTest_2)

                del dTrain, dTest, yTrain, yTest, loaded_model
                # break

            yPred = np.array([classes[int(i)]  for i in yPred])
            print([yReal, yPred, sReal])
            print('*'*30)
            true_match = (x == y for x, y in zip(yReal, yPred))
            print(np.sum(true_match)/len(yReal))
            y_accuracy = np.sum(true_match)/len(yReal)
            #s_accuracy = np.sum((x == y for x, y in zip(sReal, yPred)))/len(yReal)
            output = output + 'Y accuracy: '+str(y_accuracy)+'/n'
            #output = output + 'Y accuracy: '+str(y_accuracy)+' S accuracy: '+str(s_accuracy)+'/n'

    return output


print('[+] Getting Filenames')
files = np.array(get_files(extraRandom, permutation))
YSubClass = []
for file, num in files:
    YSubClass.append(get_name_with_survey(file))
YSubClass = np.array(YSubClass)

NUMBER_OF_POINTS = 500
while NUMBER_OF_POINTS <= MAX_NUMBER_OF_POINTS:

    print("Running Experiment")
    output = experiment(files, YSubClass, subclasses, NUMBER_OF_POINTS, n_splits)
    NUMBER_OF_POINTS += step

    text_file = open("Accuracy last model.txt", "a")
    text_file.write(output)
    text_file.close()
