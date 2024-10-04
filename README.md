# vishal_avataar_project
[Open Google Colab Notebook](https://colab.research.google.com/drive/1Fk1wel9FsPWB2HbesL6y72Y-c12t1VGQ?usp=sharing)
<!DOCTYPE html>
<html>
<head>
    <title>Grounded Segment Anything with Stable Diffusion Inpainting</title>
</head>
<body>

<h1>Grounded Segment Anything with Stable Diffusion Inpainting</h1>

<p>This project demonstrates the use of three powerful models: GroundingDINO, Segment Anything Model (SAM), and Stable Diffusion Inpainting to detect, segment, and modify objects in an image based on natural language prompts.</p>

<h2>Installation</h2>
<ol>
    <li>Navigate to the project directory:
        <pre><code>%cd /content</code></pre>
    </li>
    <li>Clone the Grounded Segment Anything repository:
        <pre><code>!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything</code></pre>
    </li>
    <li>Install the required dependencies:
        <pre><code>!pip install -q -r requirements.txt</code></pre>
    </li>
    <li>Install GroundingDINO and Segment Anything models:
        <pre><code>!pip install -q .</code></pre>
    </li>
</ol>

<h2>Models Used</h2>

<h3>1. GroundingDINO</h3>
<p><strong>Purpose:</strong> GroundingDINO is used for object detection with bounding boxes based on text prompts.</p>
<p><strong>Architecture:</strong> Transformer-based architecture for capturing global context.</p>
<p><strong>How it's used in this code:</strong> It generates bounding boxes around objects (e.g., chair) based on a natural language description.</p>

<h3>2. Segment Anything Model (SAM)</h3>
<p><strong>Purpose:</strong> SAM is used for pixel-level segmentation to refine object selection.</p>
<p><strong>Architecture:</strong> Vision transformer designed for segmentation tasks.</p>
<p><strong>How it's used in this code:</strong> It takes the bounding boxes from GroundingDINO and generates accurate masks for the objects.</p>

<h3>3. Stable Diffusion (Inpainting Pipeline)</h3>
<p><strong>Purpose:</strong> Used for inpainting, which allows modifying or replacing parts of an image based on a text description.</p>
<p><strong>Architecture:</strong> Denoising diffusion model for high-quality image generation.</p>
<p><strong>How it's used in this code:</strong> Changes the appearance or orientation of masked objects (e.g., making the chair transparent).</p>

<h2>Usage</h2>

<h3>1. Load Models</h3>
<p>After installation, the models need to be loaded:</p>
<ul>
    <li><strong>GroundingDINO:</strong> Loaded from the Hugging Face model hub.</li>
    <li><strong>SAM:</strong> Download the SAM checkpoint and initialize the model.</li>
    <li><strong>Stable Diffusion Inpainting:</strong> Loaded from Hugging Face as well.</li>
</ul>

<h3>2. Inference</h3>

<h4>Load an Image:</h4>
<p>Use the following function to download an image and load it for processing:</p>
<pre><code>def download_image(url, image_file_path):
    # Downloads the image from the provided URL.
    r = requests.get(url, timeout=4.0)
    ...
</code></pre>

<h4>Object Detection (GroundingDINO):</h4>
<p>GroundingDINO predicts the bounding boxes around objects matching the text prompt:</p>
<pre><code>def detect(image, text_prompt, model):
    ...
</code></pre>

<h4>Object Segmentation (SAM):</h4>
<p>Use SAM to generate masks for the detected objects:</p>
<pre><code>def segment(image, sam_model, boxes):
    ...
</code></pre>

<h4>Inpainting with Stable Diffusion:</h4>
<p>Modify or replace objects using the Stable Diffusion Inpainting pipeline:</p>
<pre><code>def generate_image(image, mask, prompt, negative_prompt, pipe, seed):
    ...
</code></pre>

<h2>Results</h2>
<p>Finally, the detected, segmented, and inpainted images can be displayed:</p>
<pre><code>generated_image = generate_image(image_source_pil, mask_image_pil, prompt, negative_prompt, sd_pipe, seed)
generated_image.show()
</code></pre>

<h2>Credits</h2>
<ul>
    <li><a href="https://github.com/IDEA-Research/Grounded-Segment-Anything">Grounded Segment Anything Repository</a></li>
    <li><a href="https://huggingface.co">Hugging Face Models</a></li>
</ul>

</body>
</html>


