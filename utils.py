import _pickle as cPickle
import numpy as np

def get_data_and_labels(training_file="training_set_10_227.pkl", validation_file="validation_set_10_227.pkl", test_file="test_set_10_227.pkl"):
    # conacatenate training, validation and test sets to speed up batch generation in training    
    f1, f2 ,f3 = open(training_file,"rb"), open(validation_file,"rb"), open(test_file,"rb")
    training_set, validation_set, test_set = cPickle.load(f1), cPickle.load(f2), cPickle.load(f3)
    data_dimension = training_set[sorted(training_set.keys())[0]].shape[1:]
        
    full_training_set, full_validation_set, full_test_set = [np.zeros([0,] + list(data_dimension))]*3
    full_sets = [full_training_set, full_validation_set, full_test_set]
    raw_sets = [training_set, validation_set, test_set]

    for i in range(3):
        face_keys = sorted(raw_sets[i].keys())
        for actor in face_keys:
            full_sets[i] = np.concatenate((full_sets[i], raw_sets[i][actor]))
        
    # Prepare labels
    # the data size for each actor has to be equal across all 3 data sets
    num_classes = len(training_set)
    training_size = training_set[sorted(training_set.keys())[0]].shape[0]
    validation_size = validation_set[sorted(validation_set.keys())[0]].shape[0]
    test_size = test_set[sorted(test_set.keys())[0]].shape[0]

    labels = np.eye(num_classes)
    training_labels = np.tile(labels, (1,training_size)).reshape((training_size*num_classes, -1))
    validation_labels = np.tile(labels, (1,validation_size)).reshape((validation_size*num_classes, -1))
    test_labels = np.tile(labels, (1,test_size)).reshape((test_size*num_classes, -1))
    label_sets = [training_labels, validation_labels, test_labels]
    
    return full_sets, label_sets


def get_actor_code(data_set):
    face_keys = sorted(data_set.keys())
    actor_code = {}
    reversed_actor_code = {}
    i = 0
    for actor in face_keys:
        actor_code[i] = actor
        reversed_actor_code[actor] = i
        i += 1
    return actor_code, reversed_actor_code


def get_batch(data_set, label_set, size, num_classes):
    size_per_class = size//num_classes
    indices = np.zeros((0, size_per_class), dtype=np.int)
    for i in range(num_classes):
        indices = np.append(indices,np.random.choice(range(i*data_set.shape[0]//num_classes, (i+1)*data_set.shape[0]//num_classes), size_per_class, replace=False))
    
    # indices2 = np.random.choice(data_set.shape[0], size, replace=False)
    # indices = np.append(indices, indices2)
    
    return data_set[indices], label_set[indices] 

# def get_random_batch(data_set, label_set, size):
#     indices = np.random.choice(data_set.shape[0], size, replace=False)
#     return data_set[indices], label_set[indices] 


    
    