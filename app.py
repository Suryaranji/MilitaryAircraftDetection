
from ultralytics import YOLO
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path
def video(uploaded_file,model):
    st.title("Video Processing")

    # File uploader

    if uploaded_file is not None:
        temp_file_path = os.path.join(tempfile.gettempdir(), "input_video.mp4")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Read video from the temporary file
        video = cv2.VideoCapture(temp_file_path)

        # Get video properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter("output_video.mp4", fourcc, fps, (frame_width, frame_height))

        # Process frames and write to output video
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = process_frame(frame,model)

            # Write processed frame to output video
            output_video.write(processed_frame)

        # Release video resources
        video.release()
        output_video.release()

        st.success("Video processing complete! Download the processed video below:")
        st.download_button(label="Download Processed Video", data=open("output_video.mp4", "rb").read(), file_name="output_video.mp4")
# Function to process each frame
def process_frame(frame,model):
    pred = model.predict(frame)[0].plot()
    
    #processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return pred

def predictImage(img, model):
    pred = model.predict(img)[0].plot()
    return pred

def main():
   
    with st.sidebar:
        st.title("About:")
        st.markdown(
            "- Detection of Military Aircraft using Object Detection.\n"\
            "- If you want to identify the aircrafts in an image , you can upload it here.\n"\
            "- Video Detection also there You can just upload a video and download aircraft detected video as well"
        )

           
    st.title("Militiary Aircraft Detector")
    path="runs/detect/yolov8s_100_epochs/weights/best.pt"
    model=YOLO(path)
    file = st.file_uploader("Upload a file", type=["jpg", "jpeg", "png", "gif", "mp4"])
    button = st.button("Submit")

    if button:
        st.snow()
        
        if file is None:
            st.error("❌Please upload a video or image.")
        else:     
            mime_type = file.type
            if  "image" in mime_type:
                img = plt.imread(file)
                pred = predictImage(img, model)
                st.write("Detected Image:")
                st.image(pred,width=800,channels="RGB")
            elif "video" in mime_type:
              video(file,model)

    examples=list(Path("sampleImages").glob("*.jpg"))

    # Display "Sample" on top
    st.write("## Sample Images")

    # Display sample images and their names
    num_cols = 3
    num_images = len(examples)
    num_rows = (num_images + num_cols - 1) // num_cols

    for row in range(num_rows):
        col1, col2, col3 = st.columns(3)
        for col, image_path in zip((col1, col2, col3), examples[row * num_cols: (row + 1) * num_cols]):
            with col:
                st.image(str(image_path), width=200,caption=image_path.stem)
                #st.text()  # Display filename without extension
    st.write("## Detected Video")
    st.video('newout.mp4')
            
           
if __name__ == "__main__":
    main()
