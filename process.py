import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cdist

BOW_FILE_PICKLE = "model/bow_dictionary.pkl"
SCALER_WS_FILE_PICKLE = "model/scaler_with_sift.pkl"
SVM_WS_FILE_PICKLE = "model/svm_with_sift_model.pkl"

hanacaraka = ('ba','ca','da','dha','ga','ha','ja','ka','la','ma','na','nga','nya','pa','ra','sa','ta','tha','wa','ya')

# Load Pickle File
def load_file_pickle(filename):
    file_pickle = pickle.load(open(filename, 'rb'))
    return file_pickle

# Load Image File ==================================================================================
def import_image(file) :
    image = cv2.imread(file)
    image = cv2.bitwise_not(image)
    return image
# ============================================================================================

# Preprocessing Image ========================================================================
def equalizing(img):
    img = cv2.equalizeHist(img)
    return img

def grayscaling(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def resizing(image, size):
    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
    return image

def prep_image(image):
    img = resizing(image, 192)
    img = grayscaling(img)
    img = equalizing(img)
    return img
# ============================================================================================

# Feature Extraction =========================================================================
def extract_sift_descriptor(image):
    sift = cv2.SIFT_create()
    _, descriptor = sift.detectAndCompute(image, None)
    return descriptor

def create_feature_bow(image_descriptor, bow, num_cluster):
    features = np.array([0] * num_cluster, dtype=float)

    if image_descriptor is not None:
        distance = cdist(image_descriptor, bow)
        argmin = np.argmin(distance, axis = 1)
        
        for j in argmin:
            features[j] += 1.0

    return np.array(features)

def extract_feature(image):
    img_descriptor = extract_sift_descriptor(image)
    
    num_cluster = 500
    bow = load_file_pickle(BOW_FILE_PICKLE)
    
    img_feature = create_feature_bow(img_descriptor, bow, num_cluster)
    return img_feature
# ============================================================================================

# Prediction Process ======================================================================
def predict_process(filepath):
    # Load file
    img = import_image(filepath)
    
    # Preprocessing Image
    img = prep_image(img)
    
    # Feature Extraction
    img_feature = extract_feature(img)

    # Feature Scaling
    scaler = load_file_pickle(SCALER_WS_FILE_PICKLE)
    feature_scale = scaler.transform([img_feature])
    
    # Predict SVM
    svm_model = load_file_pickle(SVM_WS_FILE_PICKLE)
    result_predict = svm_model.predict_proba(feature_scale)
    result_label = hanacaraka[result_predict.argmax()]
    result_accuracy = round(result_predict.max() * 100, 2)
    
    return result_label, result_accuracy
# ============================================================================================
