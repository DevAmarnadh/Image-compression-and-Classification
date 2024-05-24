# app.py
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from io import BytesIO
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import os
from scripts.utils import load_image_into_array
from scripts.pca_compression import PCACompressor
from scripts.utils import load_array_into_image
from scripts.utils import resize_image
import tempfile
import tensorflow as tf

app = Flask(__name__)

# Define A4 dimensions (Width x Height in millimeters)
A4 = (210, 297)

# Load the pre-trained image classification model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

def classify_image(image):
    # Preprocess the image to fit the model's input requirements
    image = image.resize((224, 224))  # MobileNetV2 input shape
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    # Predict the class probabilities
    predictions = model.predict(image)
    
    # Decode the predictions
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

def generate_pdf(image_buffer, pdf_filename):
    # Save the BytesIO object as a temporary image file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_buffer.getvalue())

    try:
        # Create a PDF with the image
        with open(pdf_filename, 'wb') as pdf_file:
            pdf = canvas.Canvas(pdf_file, pagesize=A4)
            pdf.drawImage(tmp_file.name, 0, 0, width=A4[0], height=A4[1])
            pdf.showPage()
            pdf.save()
    finally:
        # Clean up the temporary image file
        os.unlink(tmp_file.name)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image compression
@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    quality = request.form.get('quality', default=85, type=int)
    number_of_components = request.form.get('number_of_components', default=5000, type=int)

    # Calculate file sizes
    original_size_bytes = len(file.read())
    original_size_kb = original_size_bytes / 1024  # Convert to KB

    # Reset the file pointer before reading again
    file.seek(0)

    image_array = load_image_into_array(file)
    if image_array is None:
        return "Invalid image file. Please try again."

    compressor = PCACompressor(image_array)
    compressed_array = compressor.compress(number_of_components)
    compressed_image = load_array_into_image(compressed_array)

    # Save the compressed image to the server with the original image's dimensions
    compressed_image_filename = f"{file.filename}_pca_compressed.jpg"
    compressed_image.save(compressed_image_filename, format="JPEG", quality=quality)

    # Save the compressed image to the buffer for PDF generation
    buffer_for_pdf = BytesIO()
    compressed_image.save(buffer_for_pdf, format="JPEG", quality=quality)

    # Generate PDF with the compressed image
    pdf_filename = f"{file.filename}_pca_compressed.pdf"
    generate_pdf(buffer_for_pdf, pdf_filename)

    # Example of how to pass original_info to the result route
    original_info = compressor.get_original_info()

    # Calculate compressed image size in KB
    compressed_size_kb = len(buffer_for_pdf.getvalue()) / 1024  # Convert to KB

    # Classify the image
    classifications = classify_image(compressed_image)

    return render_template(
        'result.html',
        original_info=original_info,
        compressed_info=compressor.get_compressed_info(),
        original_size_kb=original_size_kb,
        compressed_size_kb=compressed_size_kb,
        compressed_image_filename=compressed_image_filename,
        compressed_pdf_filename=pdf_filename,
        classifications=[(prediction[1], prediction[2]) for prediction in classifications]
    )

# Route for converting adjusted image to PDF
@app.route('/convert_to_pdf', methods=['POST'])
def convert_to_pdf():
    adjusted_image_filename = request.form.get('adjusted_image_filename')

    if not adjusted_image_filename:
        return "Error: No filename provided."

    if not os.path.exists(adjusted_image_filename):
        return f"Error: File not found at {adjusted_image_filename}"

    try:
        # Read the adjusted image content
        adjusted_image = Image.open(adjusted_image_filename)

        # Generate the PDF filename
        pdf_filename = adjusted_image_filename.replace(".jpg", ".pdf")

        # Create a PDF with the adjusted image
        with open(pdf_filename, 'wb') as pdf_file:
            c = canvas.Canvas(pdf_file, pagesize=A4)
            c.drawImage(adjusted_image_filename, 0, 0, width=A4[0], height=A4[1])
            c.showPage()
            c.save()

        # Provide a link to download the generated PDF
        return redirect(url_for('download', filename=pdf_filename))
    except Exception as e:
        return f"Error creating PDF: {str(e)}"

# Route for downloading files
@app.route('/download/<path:filename>')
def download(filename):
    directory = os.path.dirname(filename)
    return send_from_directory(directory, os.path.basename(filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
