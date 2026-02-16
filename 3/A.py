import os
import cv2 as cv
import numpy as np
import sys

# Ρύθμιση
VOCABULARY_SIZES = [50, 100, 200, 400, 800, 10000] # Αφαιρέστε το 10000
K_VALUES = []
K_VALUES_1 = [1, 3, 5, 7, 9]
# K_VALUES_2 = [44,50]

# Επιλογή dataset (1 or 2)
DATASET_CHOICE = 1

if DATASET_CHOICE == 1:
    # Dataset 1: Caltech-Transportation
    target_path = './Data/Dataset 1'
    train_dir = './Dataset 1/caltech-transportation_train'
    test_dir = './Dataset 1/caltech-transportation_test'
    K_VALUES = K_VALUES_1
else:
    # Dataset 2: GTSRB
    target_path = './Data/Dataset 2'
    train_dir = './Dataset 2/train_modified'
    test_dir = './Dataset 2/test_modified'
    # K_VALUES = K_VALUES_2
    K_VALUES = K_VALUES_1

def create_npy_files(target_path, vocab_size=50):
    print(f'Creating npy files with vocabulary size: {vocab_size}...')
    train_categories = [i for i in os.listdir(train_dir)]

    sift = cv.xfeatures2d_SIFT.create()

    def extract_local_features(path):
        img = cv.imread(path)
        kp = sift.detect(img)
        desc = sift.compute(img, kp)
        desc = desc[1]
        return desc

    # Extract Database
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for folder in train_categories:
        files = os.listdir(os.path.join(train_dir, folder))
        for file in files:
            path = os.path.join(train_dir, folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)

    # Create vocabulary
    print(f'Creating vocabulary with {vocab_size} clusters...')
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    trainer = cv.BOWKMeansTrainer(vocab_size, term_crit, 1, cv.KMEANS_PP_CENTERS)
    vocabulary = trainer.cluster(train_descs.astype(np.float32))

    np.save(os.path.join(target_path, f'vocabulary_{vocab_size}.npy'), vocabulary)

    print('Creating index...')
    # Classification
    descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
    descriptor_extractor.setVocabulary(vocabulary)

    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in train_categories:
        files = os.listdir(os.path.join(train_dir, folder))
        for file in files:
            path = os.path.join(train_dir, folder, file)

            img = cv.imread(path)
            if img is None:
                print(f"[!] Cannot read image: {path}")
                continue
            
            # Detect keypoints
            kp = sift.detect(img)
            
            # Αν δεν βρεθούν keypoints, δοκίμασε dense sampling
            if len(kp) == 0:
                # Dense SIFT: δημιουργία keypoints σε grid
                step = 8  # pixels between keypoints
                h, w = img.shape[:2]
                kp = [cv.KeyPoint(x, y, step) 
                      for y in range(step, h - step, step)
                      for x in range(step, w - step, step)]
                if len(kp) == 0:
                    print(f"[!] No keypoints even with dense sampling: {path}")
                    continue
            
            # Compute BOW descriptor
            bow_desc = descriptor_extractor.compute(img, kp)
            
            # Αν αποτύχει το BOW, δοκίμασε με raw SIFT και manual assignment
            if bow_desc is None:
                _, descs = sift.compute(img, kp)
                if descs is not None and len(descs) > 0:
                    # Manual BOW: assign each descriptor to nearest visual word
                    bow_desc = np.zeros((1, vocabulary.shape[0]), dtype=np.float32)
                    for desc in descs:
                        dists = np.linalg.norm(vocabulary - desc, axis=1)
                        nearest = np.argmin(dists)
                        bow_desc[0, nearest] += 1
                    # L2 normalize
                    norm = np.linalg.norm(bow_desc)
                    if norm > 0:
                        bow_desc = bow_desc / norm
                else:
                    print(f"[!] Cannot compute descriptors: {path}")
                    continue

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

    np.save(os.path.join(target_path, f'index_{vocab_size}.npy'), bow_descs)
    np.save(os.path.join(target_path, f'paths_{vocab_size}.npy'), img_paths)

os.makedirs(target_path, exist_ok=True)

# Experiment with different vocabulary sizes
print(f"\n{'='*60}")
print(f"Experimenting with vocabulary sizes: {VOCABULARY_SIZES}")
print(f"Dataset: {'Caltech-Transportation' if DATASET_CHOICE == 1 else 'GTSRB'}")
print(f"{'='*60}\n")

for VOCAB_SIZE in VOCABULARY_SIZES:
    vocab_file = os.path.join(target_path, f'vocabulary_{VOCAB_SIZE}.npy')
    if not os.path.exists(vocab_file):
        print(f"\n[*] Creating files for vocabulary size: {VOCAB_SIZE}")
        create_npy_files(target_path, vocab_size=VOCAB_SIZE)
    else:
        print(f"[✓] Files for vocabulary size {VOCAB_SIZE} already exist.")

# Use the first vocabulary size for training (you can loop through all later)
CURRENT_VOCAB_SIZE = VOCABULARY_SIZES[0]
print(f"\n[*] Training classifiers with vocabulary size: {CURRENT_VOCAB_SIZE}\n")

bow_descs = np.load(os.path.join(target_path, f'index_{CURRENT_VOCAB_SIZE}.npy')).astype(np.float32)
img_paths = np.load(os.path.join(target_path, f'paths_{CURRENT_VOCAB_SIZE}.npy'))
vocabulary = np.load(os.path.join(target_path, f'vocabulary_{CURRENT_VOCAB_SIZE}.npy'))

assert bow_descs is not None
assert bow_descs is not None
assert vocabulary is not None

unique_labels = [i.split('\\')[-2] for i in img_paths]
unique_labels = list(set(unique_labels))

category_to_id = {name: i for i, name in enumerate(unique_labels)}
all_labels = [category_to_id[p.split(os.sep)[-2]] for p in img_paths]
knn_labels = np.array(all_labels, dtype=np.float32).reshape(-1, 1)

svm_path = os.path.join(target_path, f'svm_{CURRENT_VOCAB_SIZE}')
knn_path = os.path.join(target_path, f'knn_{CURRENT_VOCAB_SIZE}.xml')

# Train to every single category
# 3b svm.setKernel(cv.ml.SVM_LINEAR)
def svm_train(class_name=unique_labels[0]):
    print(f'Training SVM for class: {class_name}...')
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)  # Linear kernel as per assignment requirements
    svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
    
    # One-vs-All strategy: current class = 1, all others = 0
    labels = []
    for p in img_paths:
        # Παίρνουμε το όνομα του φακέλου (κλάση) από το path
        # path format: .../train_modified/CLASS/image.jpg
        folder_name = os.path.basename(os.path.dirname(p))
        if folder_name == class_name:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels, np.int32)
    
    # Έλεγχος ότι υπάρχουν και θετικά και αρνητικά δείγματα
    pos_count = np.sum(labels == 1)
    neg_count = np.sum(labels == 0)
    if pos_count == 0 or neg_count == 0:
        print(f'  [!] Warning: Class {class_name} has {pos_count} positive and {neg_count} negative samples. Skipping.')
        return None

    svm.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
    svm.save(os.path.join(svm_path, class_name))
    return svm
def knn_train():
    if not os.path.exists(knn_path):
        print('Training k-NN...')
        knn = cv.ml.KNearest_create()
        knn.train(bow_descs, cv.ml.ROW_SAMPLE, knn_labels)
        print(f'[+] Saving k-NN to {knn_path}')
        knn.save(knn_path)
        return knn
    else:
        print(f'[✓] k-NN already trained at {knn_path}')
        # OpenCV 3.4.2: φόρτωση με read() από FileStorage
        knn = cv.ml.KNearest_create()
        fs = cv.FileStorage(knn_path, cv.FILE_STORAGE_READ)
        knn.read(fs.getFirstTopLevelNode())
        fs.release()
        return knn

os.makedirs(svm_path, exist_ok=True)
svm_categories = os.listdir(svm_path)
if len(svm_categories) == 0:
    for train_cat in unique_labels:
        svm_train(class_name=train_cat)
    svm_categories = unique_labels
else:
    print(f"[✓] SVM already trained at {svm_path}")
    print(f"    Categories: {svm_categories}")

knn_model = knn_train()

print(f"\n{'='*60}")
print("Training Complete! Starting Evaluation...")
print(f"{'='*60}\n")

# Initialize SIFT and descriptor extractor for testing
sift = cv.xfeatures2d_SIFT.create()
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

def svm_predict_all_classes(image):
    """Predict using One-vs-All SVM - returns class with highest confidence"""
    img = cv.imread(image)
    kp = sift.detect(img)
    
    # Fallback σε dense sampling αν δεν βρεθούν keypoints
    if len(kp) == 0:
        step = 8
        h, w = img.shape[:2]
        kp = [cv.KeyPoint(x, y, step) 
              for y in range(step, h - step, step)
              for x in range(step, w - step, step)]
    
    bow_desc = descriptor_extractor.compute(img, kp)
    
    # Manual BOW αν αποτύχει το extractor
    if bow_desc is None:
        _, descs = sift.compute(img, kp)
        if descs is not None and len(descs) > 0:
            bow_desc = np.zeros((1, vocabulary.shape[0]), dtype=np.float32)
            for desc in descs:
                dists = np.linalg.norm(vocabulary - desc, axis=1)
                nearest = np.argmin(dists)
                bow_desc[0, nearest] += 1
            norm = np.linalg.norm(bow_desc)
            if norm > 0:
                bow_desc = bow_desc / norm
    
    bow_desc = bow_desc.astype(np.float32)
    
    max_confidence = -float('inf')
    predicted_class = None
    
    for category in svm_categories:
        svm = cv.ml.SVM_load(os.path.join(svm_path, category))
        response = svm.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        confidence = -response[1][0][0]  # Negative because SVM returns negative distance
        
        if confidence > max_confidence:
            max_confidence = confidence
            predicted_class = category
    
    return predicted_class

def knn_predict(image_path, k_value=5):
    """Predict using k-NN classifier"""
    img = cv.imread(image_path)
    kp = sift.detect(img)
    
    # Fallback σε dense sampling αν δεν βρεθούν keypoints
    if len(kp) == 0:
        step = 8
        h, w = img.shape[:2]
        kp = [cv.KeyPoint(x, y, step) 
              for y in range(step, h - step, step)
              for x in range(step, w - step, step)]
    
    bow_desc = descriptor_extractor.compute(img, kp)
    
    # Manual BOW αν αποτύχει το extractor
    if bow_desc is None:
        _, descs = sift.compute(img, kp)
        if descs is not None and len(descs) > 0:
            bow_desc = np.zeros((1, vocabulary.shape[0]), dtype=np.float32)
            for desc in descs:
                dists = np.linalg.norm(vocabulary - desc, axis=1)
                nearest = np.argmin(dists)
                bow_desc[0, nearest] += 1
            norm = np.linalg.norm(bow_desc)
            if norm > 0:
                bow_desc = bow_desc / norm
    
    bow_desc = bow_desc.astype(np.float32)

    ret, results, neighbors, dist = knn_model.findNearest(bow_desc, k=k_value)

    predicted_id = int(results[0, 0])
    id_to_category = {v: k for k, v in category_to_id.items()}

    return id_to_category[predicted_id]

# Evaluate classifiers
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS - Vocabulary Size: {CURRENT_VOCAB_SIZE}")
print(f"{'='*60}\n")

# Test with different k values for k-NN
for K_VAL in K_VALUES:
    print(f"\n[*] Testing with k={K_VAL} for k-NN\n")
    
    total_images = 0
    correct_svm = 0
    correct_knn = 0
    
    # Per-class results
    class_results_svm = {}
    class_results_knn = {}
    
    for category in svm_categories:
        category_path = os.path.join(test_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} does not exist")
            continue
            
        image_list = os.listdir(category_path)
        total = len(image_list)
        total_images += total
        
        correct_svm_class = 0
        correct_knn_class = 0
        
        for image in image_list:
            image_path = os.path.join(category_path, image)
            
            # SVM prediction
            svm_pred = svm_predict_all_classes(image_path)
            if svm_pred == category:
                correct_svm_class += 1
                correct_svm += 1
            
            # k-NN prediction
            knn_pred = knn_predict(image_path, k_value=K_VAL)
            if knn_pred == category:
                correct_knn_class += 1
                correct_knn += 1
        
        # Store per-class results
        class_results_svm[category] = (correct_svm_class, total)
        class_results_knn[category] = (correct_knn_class, total)
        
        # Print per-class accuracy
        svm_acc = 100 * correct_svm_class / float(total)
        knn_acc = 100 * correct_knn_class / float(total)
        print(f"{category:20s} | SVM: {correct_svm_class:3d}/{total:3d} ({svm_acc:5.2f}%) | k-NN: {correct_knn_class:3d}/{total:3d} ({knn_acc:5.2f}%)")
    
    # Print overall accuracy
    overall_svm_acc = 100 * correct_svm / float(total_images)
    overall_knn_acc = 100 * correct_knn / float(total_images)
    
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY (k={K_VAL})")
    print(f"{'='*60}")
    print(f"SVM (One-vs-All): {correct_svm}/{total_images} = {overall_svm_acc:.2f}%")
    print(f"k-NN (k={K_VAL}):     {correct_knn}/{total_images} = {overall_knn_acc:.2f}%")
    print(f"{'='*60}\n")

print("\n[✓] Evaluation complete!")