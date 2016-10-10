import os
import numpy as np

WIDTH = 32
HEIGHT = 32
CLASSES = 46
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

NAMES_AND_IDS = [['character_36_gya', '0'], ['character_10_yna', '1'], ['character_11_taamatar', '2'], ['character_12_thaa', '3'], ['character_13_daa', '4'], ['character_14_dhaa', '5'], ['character_15_adna', '6'], ['character_16_tabala', '7'], ['character_17_tha', '8'], ['character_18_da', '9'], ['character_19_dha', '10'], ['character_1_ka', '11'], ['character_20_na', '12'], ['character_21_pa', '13'], ['character_22_pha', '14'], ['character_23_ba', '15'], ['character_24_bha', '16'], ['character_25_ma', '17'], ['character_26_yaw', '18'], ['character_27_ra', '19'], ['character_28_la', '20'], ['character_29_waw', '21'], ['character_2_kha', '22'], ['character_30_motosaw', '23'], ['character_31_petchiryakha', '24'], ['character_32_patalosaw', '25'], ['character_33_ha', '26'], ['character_34_chhya', '27'], ['character_35_tra', '28'], ['character_3_ga', '29'], ['character_4_gha', '30'], ['character_5_kna', '31'], ['character_6_cha', '32'], ['character_7_chha', '33'], ['character_8_ja', '34'], ['character_9_jha', '35'], ['digit_0', '36'], ['digit_1', '37'], ['digit_2', '38'], ['digit_3', '39'], ['digit_4', '40'], ['digit_5', '41'], ['digit_6', '42'], ['digit_7', '43'], ['digit_8', '44'], ['digit_9', '45']]

# def get_labels(data_folder):
#     '''Returns a dict whose key is the class name and value is the class id.
#     Each class is a Devanagari letter. Each subfolder is regarded as the name of the class.'''
#     sub_dirs = [x[1] for x in os.walk(data_folder)][0]
#     name_to_id = {}
#     id_to_name = {}
#     names_and_ids = []
#     for i in xrange(len(sub_dirs)):
#         names_and_ids.append([sub_dirs[i], str(i)])
#         name_to_id[sub_dirs[i]] = i
#         id_to_name[str(i)] = sub_dirs[i]
#     return names_and_ids, name_to_id, id_to_name

#NAMES_AND_IDS, LABEL_NAME_TO_ID, LABEL_ID_TO_NAME = get_labels(TRAIN_DIR)
def create_maps():
    name_to_id = {}
    id_to_name = {}
    for i in xrange(len(NAMES_AND_IDS)):
        entry = NAMES_AND_IDS[i]
        name_to_id[entry[0]] = entry[1]
        id_to_name[entry[1]] = entry[0]
    return name_to_id, id_to_name


LABEL_NAME_TO_ID, LABEL_ID_TO_NAME = create_maps()


def get_label_id(class_name):
    '''Gets class_id for given class_name'''
    return LABEL_NAME_TO_ID.get(class_name)


def get_label_name(class_id):
    '''Gets name of the class corresponding to given class id'''
    return LABEL_ID_TO_NAME.get(str(class_id))


def to_label_vector(label_id):
    vector = np.zeros(CLASSES)
    vector[int(label_id)] = 1
    return vector


def to_label_vectors(label_ids):
    label_vectors = []
    for i in xrange(len(label_ids)):
        label_vector = to_label_vector(label_ids[i])
        label_vectors.append(label_vector)
    return label_vectors