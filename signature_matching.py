import streamlit as st
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

def load_resnet_model():
    base_model = tf.keras.applications.ResNet101(weights="imagenet", include_top=False, pooling="max")
    return base_model

def preprocess_image(image, size=(224, 224)):
    try:
        resized = cv2.resize(image, size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        normalized_image = tf.keras.applications.resnet50.preprocess_input(rgb_image)
        return np.expand_dims(normalized_image, axis=0)
    
    except Exception as e:
        raise ValueError(f"Error in image preprocessing: {e}")

def extract_features(image, model):
    try:
        features = model.predict(image)
        return features
    
    except Exception as e:
        raise ValueError(f"Error in feature extraction: {e}")

def compute_cosine_similarity(features1, features2):
    try:
        similarity = cosine_similarity(features1, features2)
        return similarity[0][0] * 100 
    
    except Exception as e:
        raise ValueError(f"Error in similarity computation: {e}")

def main():
    st.title("Advanced Signature Similarity Checker with Deep Learning")
    st.write("Upload two signature images to detect and compare their similarity using deep learning.")

    uploaded_file1 = st.file_uploader("Upload the first signature image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload the second signature image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 and uploaded_file2:
        try:
            model = load_resnet_model()

            img1 = Image.open(uploaded_file1)
            img2 = Image.open(uploaded_file2)
            img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
            img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

            st.image([img1, img2], caption=["First Image", "Second Image"], width=300)

            img1_preprocessed = preprocess_image(img1_cv)
            img2_preprocessed = preprocess_image(img2_cv)

            features1 = extract_features(img1_preprocessed, model)
            features2 = extract_features(img2_preprocessed, model)

            similarity = compute_cosine_similarity(features1, features2)
            st.write(f"Deep Learning-Based Similarity Score: {similarity:.2f}%")
            
            if similarity > 80:
                st.success("Signatures are highly similar. Likely authentic.")
            elif similarity > 50:
                st.warning("Signatures are moderately similar. Verify further.")
            else:
                st.error("Signatures are dissimilar. Likely forged.")

        except UnidentifiedImageError:
            st.error("One or both uploaded files are not valid images. Please upload valid image files.")
            
        except ValueError as ve:
            st.error(f"An error occurred: {ve}")
            
        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()