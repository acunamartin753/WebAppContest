# app.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# --- Define Model Architecture (must match the training script) ---
# This is necessary to load the saved model's state dictionary.

LATENT_DIM = 20
INPUT_DIM = 28 * 28
NUM_CLASSES = 10

class ConditionalVAE(nn.Module):
    def __init__(self):
        super(ConditionalVAE, self).__init__()
        # Encoder - Not needed for generation, but part of the saved model state
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM + NUM_CLASSES, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, LATENT_DIM)
        self.fc_logvar = nn.Linear(256, LATENT_DIM)
        
        # Decoder - This is what we'll use for generation
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, INPUT_DIM), nn.Sigmoid()
        )

    def decode(self, z, y):
        inputs = torch.cat([z, y], 1)
        return self.decoder(inputs)
    
    # Other methods (encode, reparameterize, forward) are not needed for generation
    # but must exist if they were part of the original class definition during saving.
    def encode(self, x, y): pass
    def reparameterize(self, mu, logvar): pass
    def forward(self, x, y): pass


# --- Load Trained Model ---
@st.cache_resource
def load_model():
    """Loads the CVAE model from the saved state file."""
    model = ConditionalVAE()
    # Load the state dictionary. map_location is used to load on CPU if needed.
    model.load_state_dict(torch.load('model/cvae_mnist.pth', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

model = load_model()


# --- Web App Interface ---
st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator") # 

st.markdown("""
Generate synthetic MNIST-like images using a trained Conditional VAE model. 
As per the exam requirements, this model was trained from scratch on the MNIST dataset. 
It can generate different images for the same digit because it samples from a random latent space. 
""")

st.sidebar.header("Controls")
# Requirement: Allow users to select which digit (0-9) to generate. 
selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))

# Requirement: A button to trigger generation. 
generate_button = st.sidebar.button("Generate Images", type="primary")

if generate_button:
    st.subheader(f"Generated images of digit {selected_digit}")

    # Generate 5 images of the same digit. 
    num_images_to_generate = 5
    
    with torch.no_grad():
        # Prepare the conditional input (the digit label)
        label = torch.zeros(1, NUM_CLASSES)
        label[0, selected_digit] = 1

        # Generate multiple images by sampling different noise vectors
        noise_vectors = torch.randn(num_images_to_generate, LATENT_DIM)
        
        # Repeat the label for each noise vector
        labels = label.repeat(num_images_to_generate, 1)

        # Generate the images
        generated_images = model.decode(noise_vectors, labels).view(-1, 1, 28, 28)

        # Create a grid to display images and convert to a displayable format
        grid = make_grid(generated_images, nrow=num_images_to_generate, normalize=True)
        img_np = grid.permute(1, 2, 0).numpy()
        
        # Display the 5 images. 
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.imshow(img_np)
        ax.axis('off') # Hide axes
        st.pyplot(fig)
        
        st.caption("From left to right: Sample 1, Sample 2, Sample 3, Sample 4, Sample 5.")

else:
    st.info("Select a digit from the sidebar and click 'Generate Images' to start.")