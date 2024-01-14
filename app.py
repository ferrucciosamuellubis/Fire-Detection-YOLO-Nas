import streamlit as st
from pipeline import predictPipeline


st.title('Fire and Smoke detection')
st.write('Detects Fire or/and Smoke in a Photo \nPowered by YOLO-Nas medium model')

st.write('')

detect_pipeline = predictPipeline()

st.info('Fire and Smoke Detection model loaded successfully!')


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    with st.container():
        col1, col2 = st.columns([3, 3])
        col1.header('Input Image')
        col1.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        col1.text('')
        col1.text('')

        if st.button('Detect'):
            detections = detect_pipeline.detect(img_path=uploaded_file)
            detections_img = detect_pipeline.drawDetections2Image(img_path=uploaded_file, detections=detections)

            col2.header('Detections')
            col2.image(detections_img, caption='Predictions by model', use_column_width=True)

