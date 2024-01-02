
from ultralytics import YOLO
import matplotlib.pyplot as plt

import streamlit as st


def predictImage(img, model):
    pred = model.predict(img)[0].plot()
    return pred

def main():
   
    with st.sidebar:
        st.title("About:")
        st.markdown(
            "- Detection of Military Aircraft using Object Detection.\n"\
            "- If you want to identify the aircrafts in an image , you can upload it here."
        )   
    st.title("Militiary Aircraft Detector")
    path="runs/detect/yolov8s_100_epochs/weights/best.pt"
    model=YOLO(path)
    file = st.file_uploader(" Image to process:")
    button = st.button("Submit")

    if button:
        st.snow()
        if not file:
            st.error("Please upload a video or image.")

        else:
                img = plt.imread(file)
                pred = predictImage(img, model)
                st.write("Detected Image:")
                st.image(pred)
if __name__ == "__main__":
   
    main()
