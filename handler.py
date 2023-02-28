from typing import  Dict, List, Any
from diffusers import StableDiffusionPipeline
import base64
import io
import torch
import logging

logger = logging.getLogger(__name__)


def encode_image(image):
  # BytesIO is a file-like buffer stored in memory
  imgByteArr = io.BytesIO()
  # image.save expects a file-like as a argument
  image.save(imgByteArr, format='JPEG')
  # Turn the BytesIO object back into a bytes object
  imgByteArr = imgByteArr.getvalue()
  im_b64 = base64.b64encode(imgByteArr).decode("utf8")
  return im_b64
    
class EndpointHandler():
    def __init__(self, path=""):
        self.pipe = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16
        )
        self.pipe.enable_attention_slicing()
        self.pipe = self.pipe.to("cuda")


    def __call__(self, data: Any) -> Dict[str, List[str]]:
        """
        :param data: A dictionary contains `inputs` and optional `parameters` field.
        :return: A dictionary with `image` field contains image in base64.
        """
        logger.info(data)
        inputs = data.pop("inputs", data)
        # hyperparamters
        num_images_per_prompt = data.pop("number", 1)
        num_inference_steps = data.pop("num_inference_steps", 50)
        guidance_scale = data.pop("guidance_scale", 7.5)
        negative_prompt = data.pop("negative_prompt", None)
        height = data.pop("height", 768)
        width = data.pop("width", 768)
        
        # run inference pipeline
        images = self.pipe(inputs, 
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_images_per_prompt,
                        negative_prompt=negative_prompt,
                        height=height,
                        width=width
        )['images']

#        return {"images":[encode_image(img) for img in images]}
        return images[0]