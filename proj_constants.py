import os
WIDTH = 32
HEIGHT = 32
CLASSES = 46
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


def get_labels(data_folder):
    '''Returns a dict whose key is the class name and value is the class id.
    Each class is a Devanagari letter. Each subfolder is regarded as the name of the class.'''
    sub_dirs = [x[1] for x in os.walk(data_folder)][0]
    name_to_id = {}
    id_to_name = {}
    for i in xrange(len(sub_dirs)):
        name_to_id[sub_dirs[i]] = i
        id_to_name[str(i)] = sub_dirs[i]
    return name_to_id, id_to_name

LABEL_NAME_TO_ID, LABEL_ID_TO_NAME = get_labels(TRAIN_DIR)


def get_label_id(class_name):
    '''Gets class_id for given class_name'''
    return LABEL_NAME_TO_ID.get(class_name)


def get_label_name(class_id):
    '''Gets name of the class corresponding to given class id'''
    return LABEL_ID_TO_NAME.get(class_id)
