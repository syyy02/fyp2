import cv2
import streamlit as st
import streamlit_option_menu
import numpy as np
from ultralytics import YOLO
from PIL import Image, ExifTags
from tensorflow.keras.models import load_model
import pandas as pd


st.set_page_config(layout="wide")
with st.sidebar:
    selected = streamlit_option_menu.option_menu(menu_title="Main Menu", options =["Home", "Intraoral Assessment", "Extraoral Assessment"], menu_icon="cast", default_index=0)

if selected == "Home":
    # Banner Image
    #st.image("C:\\Users\\user\\Downloads\\home.jpg", user_column_width =True)
    col1, col2 ,col3= st.columns([1,3,1])
    with col2:
        st.image("C:\\Users\\user\\Downloads\\home.jpg", use_column_width=True)

    # Styling with Markdown
    st.markdown("""
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
        """, unsafe_allow_html=True)

if selected == "Intraoral Assessment":
    #sub_intraselected = streamlit_option_menu.option_menu(menu_title="", options =[ "Tooth Class Classification", "Angle's Classification"], orientation="horizontal", menu_icon="cast", default_index=0)
    sub_intraselected= st.tabs(["Classification of Teeth", "Angle's Classification"])
    #if 'classification_tooth_action' not in st.session_state:
        #st.session_state.classification_tooth_action = False

    #if 'angle_classification_action' not in st.session_state:
        #st.session_state.angle_classification_action = False

    with sub_intraselected[0]:
        st.title("Classification of Teeth")
        #st.session_state.classification_tooth_action = True
        #st.session_state.angle_classification_action = False

        # Load the YOLO model with the custom weights
        # model = YOLO("C:\\Users\\user\\Downloads\\best.pt")
        #model = YOLO("C:\\Users\\user\\Documents\\fyp_model\\teethtype_best.pt")
        model = YOLO("model/intraoral_final_best.pt")

        # Define the class names manually from your data.yaml
        class_names = ["canine","incisor","molar","premolar"]
        class_colors = {
            "canine": (255, 0, 0),  # Red
            "incisor": (0, 128, 0),  # Green
            "molar": (0, 0, 255),  # Blue
            "premolar": (0, 255, 255)
        }




        # Upload image
        intra_uploaded_file = st.file_uploader("Choose a frontal view of intraoral image:", type=["jpg", "jpeg", "png"])
        #if intra_uploaded_file is not None:
        if intra_uploaded_file:
            intra_on = st.toggle("Show image")
            if intra_on:
                # Load the image and display the original
                intra_image = Image.open(intra_uploaded_file)
                #st.image(intra_image, caption="Uploaded Image", use_column_width=True)
                st.image(intra_image, caption="Uploaded Image", width=500)
        if st.button("Classify Tooth"):
            if intra_uploaded_file is not None:
                intra_image = Image.open(intra_uploaded_file)
                image_cv = np.array(intra_image.convert("RGB"))  # Convert PIL image to numpy array (OpenCV format)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

                # Run YOLOv8 model on the image
                results = model(intra_image)  # Pass the PIL image directly

                # Check if there are any detections
                detected = False  # Flag to track if teeth are detected

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        # Check if the detected class matches the tooth classes
                        if class_id < len(class_names):
                            detected = True  # At least one tooth detected
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # Box coordinates
                            label = class_names[class_id]  # Get the class label
                            label_text = f"{label}"

                            # Get the color for the class
                            color = class_colors.get(label, (255, 255, 255))

                            # Draw bounding box and label
                            # Draw the bounding box
                            # Draw the bounding box
                            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

                            # Get text size
                            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                           fontScale=1.5, thickness=2)

                            # Ensure text fits inside the box
                            text_y = y_min + text_height + 5 if y_min + text_height + 5 < y_max else y_min - 5




                            # Put the text inside the bounding box
                            cv2.putText(image_cv,
                                        label_text,
                                        (x_min, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=1.5,  # Smaller font
                                        color=color,  # Use your defined color
                                        thickness=2)

                if detected:
                    # Convert image back to RGB format for Streamlit display
                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="Predicted Image with Bounding Boxes", use_column_width=True)
                else:
                    # Display a warning if no teeth are detected
                    st.warning(
                        "⚠️ No teeth detected (incisor, canine, molar, or premolar). Please upload a valid image.")
            else:
                st.warning("⚠️ Please upload a frontal view of the intraoral image to perform classification.")

    with sub_intraselected[1]:
        # Streamlit app title
        st.title("Angle's Classification")
        st.session_state.angle_classification_action = True
        st.session_state.classification_tooth_action = False

        # Load the YOLO model with the custom weights
        # model = YOLO("C:\\Users\\user\\Downloads\\best.pt")
        #model = YOLO("C:\\Users\\user\\Downloads\\result_yolov8l_2x_overfit\\detect\\train2\\\weights\\best.pt")
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
            # Load the image and display the original
            image = Image.open(uploaded_file)
            in_on = st.toggle("Show picture")
            if in_on:
                # Load the image and display the original
                #in_image = Image.open(uploaded_file)
                #st.image(image, caption="Uploaded Intraoral Image", use_column_width=True)
                icol1, icol2, icol3 = st.columns([2, 3, 1])
                with icol2:
                    st.image(image, caption="Uploaded Intraoral Image", width=600)

            #st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            if uploaded_file is None:
                st.warning("⚠️ Please upload a side view of the intraoral image to perform classification.")
            else:

                # Convert image to numpy array for model input
                image_np = np.array(image)
                # Resize and normalize the image for YOLO model
                image_resized = cv2.resize(image_np, (640, 640))  # Use the YOLOv8 input size
                image_resized = image_resized / 255.0  # Normalize pixel values to [0, 1]

                # Run YOLO detection

                results = model(image_np)

                # Initialize to track the highest confidence for Canine and Molar
                detected_classes = {
                    "Canine": None,
                    "Molar": None
                }
                max_confidence_canine = 0
                max_confidence_molar = 0

                # Create a copy of the original image for drawing bounding boxes
                detected_image = image_np.copy()

                # Check if any boxes are detected, otherwise display message
                if results[0].boxes.data.shape[0] == 0:
                    st.warning("⚠️ Please upload a side view of the intraoral image to perform classification.")
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
                                best_canine_box = (x1, y1, x2, y2)  # Store the box coordinates


                        if "molar" in class_name.lower():
                            if confidence > max_confidence_molar:
                                max_confidence_molar = confidence
                                detected_classes["Molar"] = class_name
                                best_molar_box = (x1, y1, x2, y2)  # Store the box coordinates

                    if best_canine_box:
                        # Get text and calculate its size
                        text = f"{detected_classes['Canine']} ({max_confidence_canine:.2f})"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,  thickness=1)
                        text_width, text_height = text_size[0]

                        # Adjust text position if it exceeds the image boundary
                        x1, y1, x2, y2 = best_canine_box
                        text_x = max(0, x1)  # Ensure the text stays within the left boundary
                        text_y = max(0, y1 - 10)  # Place above the bounding box

                        if text_x + text_width > detected_image.shape[1]:  # Check if text exceeds right boundary
                            text_x = detected_image.shape[1] - text_width - 5  # Adjust to fit within the right boundary

                        # Draw text and rectangle
                        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue box
                        cv2.putText(detected_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                                    2)

                    if best_molar_box:
                        # Get text and calculate its size
                        text = f"{detected_classes['Molar']} ({max_confidence_molar:.2f})"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0, thickness=1)
                        text_width, text_height = text_size[0]

                        # Adjust text position if it exceeds the image boundary
                        x1, y1, x2, y2 = best_molar_box
                        text_x = max(0, x1)  # Ensure the text stays within the left boundary
                        text_y = max(0, y1 - 10)  # Place above the bounding box

                        if text_x + text_width > detected_image.shape[1]:  # Check if text exceeds right boundary
                            text_x = detected_image.shape[1] - text_width - 5  # Adjust to fit within the right boundary

                        # Draw text and rectangle
                        cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                        cv2.putText(detected_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                    2)

                        # Convert detected image to RGB (if necessary) and display it
                        # detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
                        rcol1, rcol2, rcol3 = st.columns([2, 3, 1])
                        with rcol2:
                            st.image(detected_image, caption="Detected Image with Bounding Boxes", width=500)
                        #st.image(detected_image, caption="Detected Image with Bounding Boxes", use_column_width=True)

                    # Output results


                    # Display detected classes with improved styling
                    st.markdown("## 🦷 **Detected Classes：**")

                    if detected_classes["Canine"] is None and detected_classes["Molar"] is None:
                        st.warning("⚠️ **No canine and molar detected.** Please upload a valid image.")
                    else:
                        if detected_classes["Canine"]:
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
                                unsafe_allow_html=True
                            )

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
                            st.markdown(
                                "<span style='color: #FF4B4B; font-weight: bold;'>❌ Molar: Not detected</span>",
                                unsafe_allow_html=True
                            )

if selected == "Extraoral Assessment":
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
        image = np.array(image)
        image = cv2.resize(image, (224, 224))  # Resize to 224x224 for DenseNet
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    # Predict class and probabilities
    def predict_class(model, image):
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)
        extraoral_class_names = ['Class I', 'Class II', 'Class III']
        return extraoral_class_names[predicted_class[0]], predictions

    # File uploader
    uploaded_file = st.file_uploader("Upload an image of a lateral view", type=["jpg", "jpeg", "png"])
    model = load_trained_model()

    if uploaded_file is not None:
        corrected_image = load_and_correct_image(uploaded_file)
        st.toast("Classifying...")

        if corrected_image is not None:
            # Display uploaded image
            ecol1, ecol2, ecol3 = st.columns([2, 3, 1])
            with ecol2:
                st.image(corrected_image, caption="Uploaded Image", width=500)



                # Preprocess the image
            image = preprocess_image(corrected_image)

            if image is not None:
                # Get predictions
                predicted_class, predictions = predict_class(model, image)

                # Display the predicted class
                st.markdown(
                    f"<h3 style='text-align: center;'>Predicted Class: <span style='color: green;'>{predicted_class}</span></h3>",
                    unsafe_allow_html=True,
                )

                # Visualize probabilities
                st.markdown("<h4 style='text-align: center;'>Prediction Probabilities:</h4>", unsafe_allow_html=True)
                extraoral_class_names = ['Class I', 'Class II', 'Class III']
                probabilities = {class_name: prob for class_name, prob in zip(extraoral_class_names, predictions[0])}

                # Display as a bar chart
                st.bar_chart(pd.DataFrame(probabilities.values(), index=probabilities.keys(), columns=["Probability"]))
                # Display plain text for probabilities
                st.markdown("<h4 style='text-align: left;'>Probability Breakdown:</h4>", unsafe_allow_html=True)
                for class_name, probability in probabilities.items():
                    st.markdown(f"- **{class_name}**: {probability:.2f}")



















