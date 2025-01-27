import streamlit as st
import streamlit_option_menu
import numpy as np
from ultralytics import YOLO
from PIL import Image, ExifTags, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import pandas as pd
import time

# Set the page configuration
st.set_page_config(layout="wide")

# Initialize session state for page selection
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

# Sidebar menu
with st.sidebar:
    selected = streamlit_option_menu.option_menu(
        menu_title="Main Menu",
        options=["Home", "Intraoral Classification", "Extraoral Classification"],
        menu_icon="cast",
        default_index=0,
    )
    st.session_state.selected_page = selected

main_placeholder = st.empty()
if st.session_state.selected_page == "Home":
    main_placeholder.empty()
    with main_placeholder.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image("images/home.jpg", use_column_width=True)
    st.markdown(
        """
        <style>
            .title {
                font-size: 28px;
                font-weight: bold;
                color: #4A90E2;
                text-align: center;
                margin-bottom: 20px;
            }
            .subtitle {
                font-size: 20px;
                color: #333333;
                margin-top: 10px;
            }
            .note {
                font-size: 16px;
                font-style: italic;
                color: #555555;
                margin-top: 20px;
            }
            ol {
                font-size: 18px;
                line-height: 1.8;
                color: #444444;
            }
            li {
                margin-bottom: 10px;
            }
            hr {
                border: none;
                border-top: 2px solid #E0E0E0;
                margin-top: 20px;
                margin-bottom: 20px;
            }
        </style>
        <div class="title">Classification of Orthodontic Images Using Deep Learning</div>
        <ol>
            <li>🦷 <strong>Intraoral Classification</strong>:  
                This menu includes the following classifications:
                <ul>
                    <li>🦷 <strong>Classification of teeth</strong>:  
                        This involves the classification of intraoral images in frontal view, upper view, and lower view into incisor, canine, premolar, and molar.
                    </li>
                    <li>📐 <strong>Angle's classification</strong>:  
                        This involves the classification of molar and canine on the left and right view of intraoral images into Class I, II, III. For Class II and III, it will display along with its subclass (1/4, 1/2, 3/4, or full class).
                    </li>
                </ul>
            <li>📸 <strong>Extraoral classification</strong>:  
                This involves the classification of extraoral side view images into Class I, II, III.
            </li>
        </ol>
        <hr>
        <div class="note"><strong>Note:</strong> The system is trained using RGB images only.</div>
        """,
        unsafe_allow_html=True,
    )

elif st.session_state.selected_page == "Intraoral Classification":
    sub_intraselected = st.tabs(["Classification of Teeth", "Angle's Classification"])

    with sub_intraselected[0]:
        st.title("Classification of Teeth")

        model = YOLO("model/intraoral_final_best.pt")

        class_names = ["canine", "incisor", "molar", "premolar"]
        class_colors = {
            "canine": (255, 0, 0),  # Red
            "incisor": (0, 128, 0),  # Green
            "molar": (0, 0, 255),  # Blue
            "premolar": (0, 255, 255),
        }

        intra_uploaded_file = st.file_uploader(
            "Please upload a frontal/upper/lower view of intraoral image:", type=["jpg", "jpeg", "png"]
        )
        if intra_uploaded_file:
            intra_on = st.toggle("Show image")
            if intra_on:
                intra_image = Image.open(intra_uploaded_file)
                incol1, incol2, incol3 = st.columns([2, 3, 1])
                with incol2:
                    st.image(intra_image, caption="Uploaded Image", width=600)

        if st.button("Classify Teeth"):
            if intra_uploaded_file is not None:
                intra_image = Image.open(intra_uploaded_file).convert("RGB")
                results = model(intra_image)
                detected = False
                draw = ImageDraw.Draw(intra_image)

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        if class_id < len(class_names):
                            detected = True
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                            label = class_names[class_id]
                            color = class_colors.get(label, (255, 255, 255))

                            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
                            font_size = 40
                            font = ImageFont.truetype("font/arial.ttf", font_size)
                            draw.text((x_min + 2, y_min - 10), label, fill=color, font=font)

                if detected:
                    ocol1, ocol2, ocol3 = st.columns([2, 3, 1])
                    with ocol2:
                        st.image(intra_image, caption="Predicted Image with Bounding Boxes", width=700)
                    st.markdown(
                        """
                        <div style="text-align: center;">
                            <h3>Color Coding for Teeth Classification:</h3>
                            <p><span style="color:green"><strong>Green</strong></span>: Incisor</p>
                            <p><span style="color:red"><strong>Red</strong></span>: Canine</p>
                            <p><span style="color:cyan"><strong>Cyan</strong></span>: Premolar</p>
                            <p><span style="color:blue"><strong>Blue</strong></span>: Molar</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error("⚠️ No teeth detected (incisor, canine, molar, or premolar). Please upload a valid image.")
            else:
                st.error("⚠️ Please upload a valid frontal/upper/lower view of the intraoral image to perform classification.")


    with sub_intraselected[1]:
        # Streamlit app title
        st.title("Angle's Classification")


        # Load the YOLO model with the custom weights
        model = YOLO("model/cnm_best.pt")

        # Define the class names manually from your data.yaml
        class_names = [
            "canine class I", "canine class II 1-2", "canine class II 1-4", "canine class II 3-4",
            "canine class II full class", "canine class III 1-2", "canine class III 1-4", "canine class III 3-4",
            "canine class III full class", "molar class I", "molar class II 1-2", "molar class II 1-4",
            "molar class II 3-4", "molar class II full class", "molar class III 1-2", "molar class III 1-4",
            "molar class III 3-4", "molar class III full class"
        ]

        # Streamlit app instructions
        #st.write("Upload an side view intraoral image for classification of  molar and canine class types.")

        # Upload image
        uploaded_file = st.file_uploader("Choose a side view intraoral image:", type=["jpg", "jpeg", "png"])
        #in_on = st.toggle("Show uploaded picture")
        #if in_on:
            # Load the image and display the original
            #image = Image.open(uploaded_file)
            #st.image(image, caption="Uploaded Image", use_column_width=True)

        if uploaded_file:
            # Load the image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            in_on = st.toggle("Show picture")
            if in_on:
                icol1, icol2, icol3 = st.columns([2, 3, 1])
                with icol2:
                    st.image(image, caption="Uploaded Intraoral Image", width=600)

        if st.button("Classify"):
            if uploaded_file is None:
                st.error("⚠️ Please upload a side view of the intraoral image to perform classification.")
            else:
                # Convert image to numpy array
                image_np = np.array(image)

                # Resize and normalize the image for YOLO model
                image_resized = image.resize((640, 640))  # Resize using Pillow
                image_np_resized = np.array(image_resized) / 255.0  # Normalize pixel values

                # Run YOLO detection
                results = model(image_np)

                # Initialize to track the highest confidence for Canine and Molar
                detected_classes = {"Canine": None, "Molar": None}
                max_confidence_canine = 0
                max_confidence_molar = 0

                # Create a copy of the original image for drawing
                detected_image = image.copy()
                draw = ImageDraw.Draw(detected_image)

                # Check if any boxes are detected
                if results[0].boxes.data.shape[0] == 0:
                    st.error("⚠️ No detections. Please upload a valid image.")
                else:
                    best_canine_box = None
                    best_molar_box = None

                    for pred in results[0].boxes.data:
                        class_id = int(pred[5].item())
                        class_name = class_names[class_id].replace("-", "/")
                        confidence = pred[4].item()  # Confidence score

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, pred[:4])  # Get box coordinates

                        if "canine" in class_name.lower():
                            if confidence > max_confidence_canine:
                                max_confidence_canine = confidence
                                detected_classes["Canine"] = class_name
                                best_canine_box = (x1, y1, x2, y2)

                        if "molar" in class_name.lower():
                            if confidence > max_confidence_molar:
                                max_confidence_molar = confidence
                                detected_classes["Molar"] = class_name
                                best_molar_box = (x1, y1, x2, y2)

                    # Draw bounding boxes and text using Pillow
                    #font = ImageFont.load_default()
                    font_size = 40
                    font = ImageFont.truetype("font/arial.ttf", font_size)

                    if best_canine_box:
                        x1, y1, x2, y2 = best_canine_box
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)  # Red box
                        draw.text((x1, y1 - 10), f"{detected_classes['Canine']} ",
                                  fill="green", font=font)

                    if best_molar_box:
                        x1, y1, x2, y2 = best_molar_box
                        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)  # Green box
                        draw.text((x1, y1 - 10), f"{detected_classes['Molar']}",
                                  fill="yellow", font=font)


                    # Display the image with bounding boxes
                    rcol1, rcol2, rcol3 = st.columns([2, 3, 1])
                    with rcol2:
                        st.image(detected_image, caption="Detected Image with Bounding Boxes", width=600)
                        #st.image(enlarged_image, caption="Detected Image with Bounding Boxes", use_column_width=True)

                # Output detected classes with styling
                #st.markdown("## 🦷 **Detected Classes：**")
                    if detected_classes["Canine"] is None and detected_classes["Molar"] is None:
                        st.error("⚠️ **No canine and molar detected.** Please upload a valid image.")
                    else:
                        if detected_classes["Canine"]:
                            st.markdown("## 🦷 **Detected Classes：**")
                            st.markdown(
                                f"""
                                <div style='background-color: #DFF6FF; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                                    <b>🟢 Canine Detected:</b> {detected_classes['Canine']}<br>
                                    <b>Confidence:</b> {max_confidence_canine:.2f}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                "<span style='color: #FF4B4B; font-weight: bold;'>❌ Canine: Not detected</span>",
                                unsafe_allow_html=True)

                        if detected_classes["Molar"]:
                            st.markdown(
                                f"""
                                <div style='background-color: #FFF7D6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>
                                    <b>🟡 Molar Detected:</b> {detected_classes['Molar']}<br>
                                    <b>Confidence:</b> {max_confidence_molar:.2f}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown("<span style='color: #FF4B4B; font-weight: bold;'>❌ Molar: Not detected</span>",
                                        unsafe_allow_html=True)


else:
    st.title("🧑‍⚕️Extraoral Orthodontic Images Classification👩‍⚕️")

    # Load the pre-trained model
    def load_trained_model():
        model = load_model('model/densenet121_zm9055_1.keras', compile=False)
        return model

    # Load and correct image orientation
    def load_and_correct_image(uploaded_file):
        try:
            image = Image.open(uploaded_file)
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation, 1)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image

        except Exception as e:
            st.error("Error loading image.")
            print("Error:", e)
            return None

    # Preprocess the image
    def preprocess_image(image):
        if image is None:
            st.error("Invalid image.")
            return None
        # Resize the image to 224x224 using Pillow
        image = image.resize((224, 224))
        image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    # Predict class and probabilities
    # Predict class and probabilities
    def predict_class(model, image, threshold=0.5):
        predictions = model.predict(image)
        max_probability = np.max(predictions)
        predicted_class_index = np.argmax(predictions, axis=1)
        extraoral_class_names = ['Class I', 'Class II', 'Class III']

        if max_probability < threshold:
            return None, predictions  # No confident class detected
        return extraoral_class_names[predicted_class_index[0]], predictions


    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a lateral view", type=["jpg", "jpeg", "png"])
    model = load_trained_model()

    if uploaded_file is not None:

        corrected_image = load_and_correct_image(uploaded_file)
        st.toast("Classifying...")
        time.sleep(.2)

        if corrected_image is not None:
            # Display uploaded image
            ecol1, ecol2, ecol3 = st.columns([2, 3, 1])
            with ecol2:
                st.image(corrected_image, caption="Uploaded Image", width=500)

            # Preprocess the image
            #corrected_image = corrected_image.convert('RGB')
            image = preprocess_image(corrected_image)

            if image is not None:

                # Get predictions
                predicted_class, predictions = predict_class(model, image)

                if predicted_class is None:
                    # Display error message for irrelevant images
                    st.error("⚠️ **No confident class detected. Please ensure the image is relevant and clear.**")
                else:
                    # Display the predicted class
                    st.markdown(
                        f"<h3 style='text-align: center;'>Predicted Class: <span style='color: green;'>{predicted_class}</span></h3>",
                        unsafe_allow_html=True,
                    )

                    # Visualize probabilities
                    st.markdown("<h4 style='text-align: center;'>Prediction Probabilities:</h4>",
                                unsafe_allow_html=True)
                    extraoral_class_names = ['Class I', 'Class II', 'Class III']
                    probabilities = {class_name: prob for class_name, prob in
                                     zip(extraoral_class_names, predictions[0])}

                    # Display as a bar chart
                    st.bar_chart(
                        pd.DataFrame(probabilities.values(), index=probabilities.keys(), columns=["Probability"]))

                    # Display plain text for probabilities
                    st.markdown("<h4 style='text-align: left;'>Probability Breakdown:</h4>", unsafe_allow_html=True)
                    for class_name, probability in probabilities.items():
                        st.markdown(f"- **{class_name}**: {probability:.2f}")























