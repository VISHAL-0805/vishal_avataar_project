# vishal_avataar_project
[Open Google Colab Notebook](https://colab.research.google.com/drive/1Fk1wel9FsPWB2HbesL6y72Y-c12t1VGQ?usp=sharing)
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grounded-Segment-Anything with Stable Diffusion Inpainting</title>
</head>
<body>

<h1>Grounded-Segment-Anything with Stable Diffusion Inpainting</h1>

<p>This project combines object detection using GroundingDINO, segmentation using Segment Anything (SAM), and inpainting with Stable Diffusion to process images and modify specific objects within a scene.</p>

<h2>Installation</h2>

<p>1. Change to the content directory:<br>
<code>%cd /content</code></p>

<p>2. Clone the Grounded-Segment-Anything repository:<br>
<code>!git clone https://github.com/IDEA-Research/Grounded-Segment-Anything</code></p>

<p>3. Navigate to the project directory and install dependencies:<br>
<code>%cd /content/Grounded-Segment-Anything<br>!pip install -q -r requirements.txt</code></p>

<p>4. Install GroundingDINO:<br>
<code>%cd /content/Grounded-Segment-Anything/GroundingDINO<br>!pip install -q .</code></p>

<p>5. Install Segment Anything:<br>
<code>%cd /content/Grounded-Segment-Anything/segment_anything<br>!pip install -q .</code></p>

<h2>Import Required Libraries</h2>

<p>In this step, we import necessary Python libraries to run the models:</p>

<pre>
<code>
import os, sys
import argparse, copy
from PIL import Image
from torchvision.ops import box_convert
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import build_sam, SamPredictor
</code>
</pre>

<h2>Load Models</h2>

<h3>GroundingDINO Model</h3>

<p>To load the GroundingDINO model, use the following function. This loads the configuration file and model weights from Hugging Face Hub:</p>

<pre>
<code>
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    args.device = device
    model = build_model(args)
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model
</code>
</pre>

<p>Set the repository details and load the model:</p>

<pre>
<code>
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
</code>
</pre>

<h3>SAM Model</h3>

<p>Download SAM's checkpoint file and load it into memory:</p>

<pre>
<code>
! wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
</code>
</pre>

<h3>Stable Diffusion for Inpainting</h3>

<p>Load the Stable Diffusion inpainting pipeline:</p>

<pre>
<code>
sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to(device)
</code>
</pre>

<h2>Inference</h2>

<p>First, download and load an image:</p>

<pre>
<code>
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)
    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))
</code>
</pre>

<p>Specify the image path and URL:</p>

<pre>
<code>
local_image_path = "/content/sample_data/chair.jpg"
image_url = "your_image_url"
download_image(image_url, local_image_path)
</code>
</pre>

<h3>Grounding DINO for Object Detection</h3>

<p>Use GroundingDINO to detect objects in the image with a text prompt:</p>

<pre>
<code>
def detect(image, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold
  )
  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  return annotated_frame, boxes
</code>
</pre>

<h3>Segmenting Detected Object Using SAM</h3>

<p>Next, we use SAM to segment the object detected by GroundingDINO:</p>

<pre>
<code>
def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()
</code>
</pre>

<h3>Inpainting Using Stable Diffusion</h3>

<p>Use Stable Diffusion for inpainting over the segmented mask:</p>

<pre>
<code>
def generate_image(image, mask, prompt, negative_prompt, pipe, seed):
  in_image = image.resize((512, 512))
  in_mask = mask.resize((512, 512))
  generator = torch.Generator(device).manual_seed(seed)
  result = pipe(image=in_image, mask_image=in_mask, prompt=prompt, negative_prompt=negative_prompt, generator=generator)
  result = result.images[0]
  return result.resize((w, h))
</code>
</pre>

<p>Provide the desired prompt and negative prompt for the inpainting task:</p>

<pre>
<code>
prompt="transparent chair"
negative_prompt="low resolution, ugly"
generated_image = generate_image(image=image_source_pil, mask=image_mask_pil, prompt=prompt, negative_prompt=negative_prompt, pipe=sd_pipe, seed=32)
</code>
</pre>

<h2>License</h2>

<p>This project is licensed under the MIT License.</p>

</body>
</html>

