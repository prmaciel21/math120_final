import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

IMG_SIZE = (64, 64)  # Resize images to 64x64

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

def compute_pca(X, n_components=50):
    '''
    Standardize the data and compute PCA.
    '''

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

def classifier(X_pca, y):
    '''
    Train and evaluate a simple k-NN classifier on the PCA-transformed data.
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    print(f"Classifier accuracy: {accuracy * 100:.2f}%")
    return accuracy