# vishal_avataar_project
[Open Google Colab Notebook](https://colab.research.google.com/drive/1Fk1wel9FsPWB2HbesL6y72Y-c12t1VGQ?usp=sharing)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grounded Segment Anything with Stable Diffusion</title>
</head>
<body>

<h1>Grounded Segment Anything with Stable Diffusion</h1>

<p>This project demonstrates how to combine the power of <b>Grounding DINO</b> for object detection, <b>Segment Anything Model (SAM)</b> for segmentation, and <b>Stable Diffusion</b> for image inpainting to manipulate images. Below is the explanation of each component and a walkthrough of the code.</p>

<h2>Steps and Code Explanation</h2>

<h3>1. Install Dependencies</h3>
<p>The project requires various libraries such as <code>GroundingDINO</code>, <code>Segment Anything</code>, <code>Supervision</code>, <code>Stable Diffusion Inpainting</code> pipeline, and other necessary dependencies. The following code installs them:</p>
<pre>
<code>
%cd /content
!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
%cd /content/Grounded-Segment-Anything
!pip install -q -r requirements.txt
</code>
</pre>

<h3>2. Import Required Libraries</h3>
<p>This section imports all necessary libraries such as <code>PIL</code> for image processing, <code>torchvision</code> for transformations, <code>GroundingDINO</code>, and <code>Segment Anything</code> models, and <code>Stable Diffusion</code> for inpainting.</p>
<pre>
<code>
import os, sys
import argparse
import copy
from IPython.display import display
from PIL import Image
from torchvision.ops import box_convert
import GroundingDINO.groundingdino.datasets.transforms as T
import supervision as sv
from segment_anything import build_sam, SamPredictor
import torch
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
</code>
</pre>

<h3>3. Define Models Used</h3>

<h4>Grounding DINO Model</h4>
<p><b>Grounding DINO</b> is used for object detection and localization based on a text prompt. We define a helper function to load the model and set up the checkpoint from Hugging Face.</p>
<pre>
<code>
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    return model

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
</code>
</pre>

<h4>Segment Anything Model (SAM)</h4>
<p><b>SAM</b> is used for segmentation. It allows us to generate masks for detected objects. The following code loads the SAM model:</p>
<pre>
<code>
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
</code>
</pre>

<h4>Stable Diffusion Inpainting Model</h4>
<p><b>Stable Diffusion</b> is used for inpainting to modify parts of the image based on a mask. The following code loads the Stable Diffusion model:</p>
<pre>
<code>
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)
</code>
</pre>

<h3>4. Loading the Input Image</h3>
<p>We download the input image from a URL and save it locally:</p>
<pre>
<code>
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)
    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)

local_image_path = "/content/sample_data/chair.jpg"
image_url = "IMAGE_URL"
download_image(image_url, local_image_path)
image_source, image = load_image(local_image_path)
Image.fromarray(image_source)
</code>
</pre>

<h3>5. Object Detection using Grounding DINO</h3>
<p>This step uses <b>Grounding DINO</b> to detect objects in the image based on a text prompt like "big chair". It returns bounding boxes and phrases for detected objects:</p>
<pre>
<code>
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(model=model, image=image, caption=text_prompt)
  return annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

annotated_frame, detected_boxes = detect(image, text_prompt="big chair", model=groundingdino_model)
Image.fromarray(annotated_frame)
</code>
</pre>

<h3>6. Segmentation using SAM</h3>
<p>We use <b>SAM</b> to segment the detected objects in the image based on the bounding boxes provided by Grounding DINO. This generates a mask for the detected object:</p>
<pre>
<code>
segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
Image.fromarray(annotated_frame_with_mask)
</code>
</pre>

<h3>7. Inpainting using Stable Diffusion</h3>
<p>Finally, we use <b>Stable Diffusion</b> to generate a new image by inpainting based on a prompt. The following function takes the original image, the mask generated by SAM, and a text prompt (e.g., "transparent chair") to modify the image:</p>
<pre>
<code>
def generate_image(image, mask, prompt, negative_prompt, pipe, seed):
  w, h = image.size
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))
  generator = torch.Generator(device).manual_seed(seed)
  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
  return result.resize((w, h))

prompt = "transparent chair"
negative_prompt = "low resolution, ugly"
seed = 32
generated_image = generate_image(image_source_pil, image_mask_pil, prompt, negative_prompt, sd_pipe, seed)
generated_image
</code>
</pre>

<h3>8. Rotating the Object</h3>
<p>We can also manipulate the object further, such as rotating it, by providing an inverse mask and generating a new image:</p>
<pre>
<code>
prompt = "left rotate chair"
negative_prompt = "people, low resolution, ugly"
generated_image = generate_image(image_source_pil, inverted_image_mask_pil, prompt, negative_prompt, sd_pipe, seed)
generated_image
</code>
</pre>

</body>
</html>
