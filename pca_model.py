import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = (96, 96)  # Resize images to 64x64

def load_img(img_dir):
    '''
    Load images from a directory, resize them, and return as a numpy array.
    img_dir/person/img.jpg
    '''

    X = []
    y = []
    label_map = {}
    label = 0

    for person in os.listdir(img_dir):
        person_dir = os.path.join(img_dir, person)
        if not os.path.isdir(person_dir):
            continue

        label_map[label] = person

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten())
            y.append(label)

        label += 1
    
    return np.array(X), np.array(y), label_map

def compute_pca(X, variance_threshold=0.95):
    '''
    Normalize data and compute PCA, selecting components
    to preserve the desired variance.
    '''

    #normalize pixel values
    X = X / 255.0

    #first PCA to determine number of components
    pca_full = PCA(whiten=True, random_state=42)
    X_pca_full = pca_full.fit_transform(X)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= variance_threshold) + 1

    print(f"Using {n_components} PCA components")

    # Final PCA
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X)

    return pca, X_pca

def plot_mean_face(X):
    '''
    Plot the mean face from the PCA model.
    '''
    mean_face = np.mean(X, axis=0)
    plt.imshow(mean_face.reshape(IMG_SIZE), cmap='gray')
    plt.title("Mean Face")
    plt.axis('off')
    plt.show()

def plot_eigenfaces(pca, n_faces=10):
    '''
    Plot the top n_faces eigenfaces from the PCA model.
    '''

    fig, axes = plt.subplots(2, n_faces//2, figsize=(12, 5))

    for i, ax in enumerate(axes.flat):
        eigenface = pca.components_[i].reshape(IMG_SIZE)
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f"Eigenface {i+1}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def classifier(X_pca, y, label_map=None, show_confusion=True):
    '''
    Train and evaluate an SVM classifier on PCA features.
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, stratify=y, random_state=42
    )

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    )

    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)

    print(f"SVM accuracy: {accuracy * 100:.2f}%")

    # Optional confusion matrix (great for Colab demo)
    if show_confusion:
        y_pred = svm.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(xticks_rotation="vertical")
        plt.title("SVM Confusion Matrix")
        plt.show()

    return svm, accuracy
