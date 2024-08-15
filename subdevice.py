from flask import Flask, jsonify, request
import socket
import cv2
import numpy as np
import os
import zipfile
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from flask_cors import CORS  # Import CORS
import matplotlib.pyplot as plt
import time
from flask import Flask, jsonify, render_template  # Add render_template here
  # Enable CORS for the entire application
# Constants
MAIN_DEVICE_IP = '192.168.69.12'  # Replace with the main device IP
PORT = 5000
BUFFER_SIZE = 4096
IMAGE_FOLDER = 'captured_images'  # Folder within the project directory
PROCESSED_FOLDER = 'processed_images'  # Folder for processed images
ZIP_FILENAME = 'processed_images.zip'
NUM_IMAGES = 10  # Number of images to capture

app = Flask(__name__)
CORS(app)


def capture_and_save_images():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Get the absolute path for the image folder
    folder_path = os.path.abspath(IMAGE_FOLDER)
    print(f"Absolute path of folder: {folder_path}")

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Creating folder...")
        try:
            os.makedirs(folder_path)  # Create the folder if it doesn't exist
            print(f"Folder {folder_path} created successfully.")
        except Exception as e:
            print(f"Error creating folder: {e}")
            cap.release()
            return None
    else:
        print(f"Folder {folder_path} already exists.")

    images_captured = 0
    for i in range(NUM_IMAGES):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture image {i + 1}.")
            continue

        # Save the image to the folder
        image_path = os.path.join(folder_path, f'image_{i + 1}.jpg')
        print(f"Saving image {i + 1} to {image_path}...")
        try:
            cv2.imwrite(image_path, frame)
            print(f"Image {i + 1} saved successfully.")
            images_captured += 1
        except Exception as e:
            print(f"Error saving image {i + 1}: {e}")
        time.sleep(0.01)

    cap.release()

    if images_captured == 0:
        print("No images were captured.")
        return None

    return folder_path  # Return the folder path if images were captured



def zip_folder(folder_path, zip_filename):
    print(f"Zipping folder {folder_path} into {zip_filename}...")
    try:
        with zipfile.ZipFile(zip_filename, 'w') as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))
        print(f"Folder {folder_path} zipped successfully into {zip_filename}.")
    except Exception as e:
        print(f"Error creating ZIP file: {e}")


def send_zip_to_main_device(zip_filename):
    print(f"Sending {zip_filename} to main device...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((MAIN_DEVICE_IP, PORT))

        # Send the zip file
        with open(zip_filename, 'rb') as f:
            while (chunk := f.read(BUFFER_SIZE)):
                client_socket.sendall(chunk)

        client_socket.close()
        print("ZIP file sent successfully.")
    except Exception as e:
        print(f"Error sending ZIP file: {e}")


# Debris classification code
class DebrisAgent(Agent):
    def _init_(self, unique_id, model, image_paths=None):
        super()._init_(unique_id, model)
        self.image_paths = image_paths
        self.results = None  # Initialize results to None

    def step(self):
        if self.results is None:  # Process images only if results are None
            best_image_path, best_results = self.select_best_image()
            self.results = best_results
            print(f"Agent {self.unique_id + 1}: Best Image - {best_image_path}")
            print(f"Agent {self.unique_id + 1}: Results - {self.results}")

    def select_best_image(self):
        best_image_path = None
        best_results = None
        max_sharpness = 0

        for image_path in self.image_paths:
            sharpness = self.calculate_laplacian_sharpness(image_path)
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_image_path = image_path
                best_results = self.classify_debris(image_path)
        return best_image_path, best_results

    def calculate_laplacian_sharpness(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        return sharpness

    def classify_debris(self, image_path):
        # Sample parameters for GSD calculation (these should be adjusted as per your scenario)
        height = 100
        sensor_width = 36
        sensor_height = 24
        focal_length = 50

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return {}

        gsd_width = calculate_gsd(height, sensor_width, image.shape[1], focal_length)
        gsd_height = calculate_gsd(height, sensor_height, image.shape[0], focal_length)
        gsd = (gsd_width + gsd_height) / 2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        overlap_counts = np.zeros_like(image[:, :, 0], dtype=int)
        for (x, y, w, h) in bounding_boxes:
            overlap_counts[y:y + h, x:x + w] += 1

        output_image = image.copy()
        total_area_70 = total_area_80 = total_area_90 = total_area_100 = 0
        count_70 = count_80 = count_90 = count_100 = 0

        for (x, y, w, h) in bounding_boxes:
            region_overlap_count = overlap_counts[y:y + h, x:x + w].max()

            if region_overlap_count >= 4:
                color = (100, 100, 255)
                total_area_100 += w * h * (gsd ** 2)
                count_100 += 1
            elif region_overlap_count == 3:
                color = (100, 200, 255)
                total_area_90 += w * h * (gsd ** 2)
                count_90 += 1
            elif region_overlap_count == 2:
                color = (100, 255, 100)
                total_area_80 += w * h * (gsd ** 2)
                count_80 += 1
            else:
                color = (255, 100, 100)
                total_area_70 += w * h * (gsd ** 2)
                count_70 += 1

            overlay = output_image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

        # Save the processed image
        processed_image_path = os.path.join(PROCESSED_FOLDER, f'processed_{self.unique_id + 1}.jpg')
        if not os.path.exists(PROCESSED_FOLDER):
            os.makedirs(PROCESSED_FOLDER)
        cv2.imwrite(processed_image_path, output_image)

        return {
            "low damage": {"regions": count_70, "total_area": total_area_70},
            "medium damage": {"regions": count_80, "total_area": total_area_80},
            "high damage": {"regions": count_90, "total_area": total_area_90},
            "severe damage": {"regions": count_100, "total_area": total_area_100}
        }


class DebrisModel(Model):
    def _init_(self, image_paths):
        super()._init_()
        self.num_agents = len(image_paths)
        self.schedule = BaseScheduler(self)
        self.datacollector = DataCollector(agent_reporters={"Results": lambda a: a.results})

        print(f"{self.num_agents} data agents are generating")
        for i, agent_image_paths in enumerate(image_paths):
            agent = DebrisAgent(i, self, agent_image_paths)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


def calculate_gsd(height, sensor_size, image_dimension, focal_length):
    return (height * sensor_size) / (image_dimension * focal_length)


@app.route('/process_images',methods=['POST'])
def process_images():
    data = request.json
    ip_address = data.get('ip_address')
    global MAIN_DEVICE_IP
    MAIN_DEVICE_IP = ip_address
    print(f"THIS is from process images{MAIN_DEVICE_IP}")
    folder_path = capture_and_save_images()
    if folder_path:
        # Prepare paths for each agent's images
        agent_image_paths = [
            [os.path.join(folder_path, f'image_{i * NUM_IMAGES + j + 1}.jpg') for j in range(NUM_IMAGES)]
            for i in range(1)]

        # Process images with DebrisModel
        print(f"Number of agents: {len(agent_image_paths)}")

        model = DebrisModel(agent_image_paths)
        model.step()  # Run the model for all agents
        data = model.datacollector.get_agent_vars_dataframe()
        output_file_path = 'model_data.txt'

        # Write DataFrame to a text file
        with open(output_file_path, 'w') as file:
            file.write(data.to_csv(index=False, header=False, sep='\t'))

        print(f"Data has been written to {output_file_path}")
        new_file_path = os.path.join(PROCESSED_FOLDER, output_file_path)
        if os.path.exists(new_file_path):
            os.remove(new_file_path)
            print(f"Existing file '{new_file_path}' deleted.")
        os.rename(output_file_path, new_file_path)
        files = os.listdir(PROCESSED_FOLDER)
        print("---------------")
        for file in files:
            print(file)
        print("---------------")
        # Zip and send the processed images
        zip_folder(PROCESSED_FOLDER, ZIP_FILENAME)
        send_zip_to_main_device(ZIP_FILENAME)

        # Optionally, clean up
        try:
            for file in os.listdir(PROCESSED_FOLDER):
                os.remove(os.path.join(PROCESSED_FOLDER, file))
            os.rmdir(PROCESSED_FOLDER)
            os.rmdir(folder_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")

        return jsonify({"status": "success", "message": "Images processed and sent successfully."})
    else:
        return jsonify({"status": "error", "message": "No images captured."})



@app.route('/')
def my_home():
    return "home"
@app.route('/startclient', methods=['POST'])
def start_server():
    try:
        data = request.json
        ip_address = data.get('ip_address')
        # You can now use the ip_address as needed
        # For example, store it in MAIN_DEVICE_IP
        global MAIN_DEVICE_IP
        MAIN_DEVICE_IP = ip_address
        print(MAIN_DEVICE_IP)
        # Add your server start logic here
        # For now, just returning a success message
        return jsonify({"message": "CLIENT is running successfully."})
    except Exception as e:
        return jsonify({"message": f"Failed to start CLIENT 7777: {e}"}), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)