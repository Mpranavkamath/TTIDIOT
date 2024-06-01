import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import customtkinter as ctk
import tkinter as tk
from PIL import Image
from authtoken import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create a tkinter app
app = tk.Tk()
app.geometry("532x622")  # Fix the geometry method call
app.title("sachin check if u have ur bus pass bro")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=500, text_color="black", fg_color="white")
prompt.place(x=16, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
  # Specify the master parameter
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):  # sends to GPU fp16 also is used for this
        result = pipe(prompt.get(), guidance_scale=8.5)
        print(result)  # Debugging: print the entire result to understand the structure

        # Check available keys and access the correct one
        if "images" in result:
            image = result["images"][0]
        else:
            raise KeyError("Expected key 'images' not found in the output.")

    # Save the image using PIL Image's save method
    image.save("generated_image.png")

    # Convert the image to a format that CTkImage can use
    ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(512, 512))
    lmain.configure(image=ctk_image)
    lmain.image = ctk_image  # Keep a reference to avoid garbage collection

trigger = ctk.CTkButton(master=app, height=40, width=120, text_color="white", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

# Main loop for the app
app.mainloop()
