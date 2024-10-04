# vishal_avataar_project
[Open Google Colab Notebook](https://colab.research.google.com/drive/1Fk1wel9FsPWB2HbesL6y72Y-c12t1VGQ?usp=sharing)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grounded-Segment-Anything Project</title>
</head>
<body>

<h1>Grounded-Segment-Anything</h1>
<p>This project integrates GroundingDINO, Segment Anything (SAM), and Stable Diffusion for object detection, segmentation, and inpainting tasks. Below are the steps to set up and use this project.</p>

<h2>Installation</h2>
<p>1. Clone the repository:</p>
<pre><code>git clone https://github.com/IDEA-Research/Grounded-Segment-Anything</code></pre>

<p>2. Navigate to the project directory:</p>
<pre><code>cd Grounded-Segment-Anything</code></pre>

<p>3. Install the dependencies:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<p>4. Install the <strong>GroundingDINO</strong> model:</p>
<pre><code>cd GroundingDINO && pip install .</code></pre>

<p>5. Install the <strong>Segment Anything</strong> model:</p>
<pre><code>cd ../segment_anything && pip install .</code></pre>

<h2>Model Loading</h2>
<p>Before you run inference, make sure you load the required models.</p>

<p>1. Load the GroundingDINO model:</p>
<pre><code>from GroundingDINO.groundingdino.models import build_model
model = build_model(args)
</code></pre>

<p>2. Download SAM checkpoints:</p>
<pre><code>wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth</code></pre>

<p>3. Load SAM model in your code:</p>
<pre><code>from segment_anything import build_sam, SamPredictor
sam_predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth").to(device))
</code></pre>

<h2>Inference</h2>
<p>Use the following code to download and preprocess an image, and then use the models for inference.</p>
<pre><code>import requests
from PIL import Image
from io import BytesIO

def download_image(url, image_file_path):
    r = requests.get(url)
    if r.status_code == 200:
        with Image.open(BytesIO(r.content)) as img:
            img.save(image_file_path)
            print(f"Image saved to: {image_file_path}")
    else:
        print("Failed to download image")

image_url = "your_image_url"
local_image_path = "sample_image.jpg"
download_image(image_url, local_image_path)
</code></pre>

<h2>Contributing</h2>
<p>Feel free to submit issues or contribute to the project. You can do this by forking the repository and submitting a pull request with your changes.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
</html>
