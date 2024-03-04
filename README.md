# Visualizing Internal Representations in a Latent Diffusion Model
Linear probes found controllable representations of scene attributes in a text-to-image diffusion model

# 🤔 Purpose
This is a source repository for the public-facing website accompaniment to the DSC180B final project.

Team: Karina Chen, Atharva Kulkarni, Ester Tsai, Zelong Wang.

This GitHub Pages uses the **[Agency](https://startbootstrap.com/theme/agency)** theme from Bootstrap. 

### Using pip environment
If you don't have virtualenv installed, run "pip install virtualenv"  
To create a new environment, run "virtualenv venv", replacing venv with whatever you want to call your environment  
To activate, run "source venv/bin/activate"  
Then install the requirments in the current environment using "pip install -r requirements.txt"

## Probe Weights:
Unzip the [probe_checkpoints.zip](https://github.com/zew013/diffusion-model-internal-probing/blob/main/probe_checkpoints.zip) to acquire all probe weights. The probe weights in the unzipped folder should be sufficient for you to run all experiments. 

## Build Instructions
This project stores all of the library code in .py files. The Jupyter Notebooks contain the build scripts and other crucial instructions.

Please use the corresponding Jupyter Notebooks for each step below.

Step 1: Generate the images and ground truth labels using `create_the_synthetic_dataset.ipynb`

Step 2: Train the probes using `run_probing_experiments.ipynb`
- Or skip this step by using the pre-trained probe weights described in the section above.

Step 3: Output the intermediate images for each model and create the comparison plots using this build script: `python create_plot.py --probe ind {probe index}`
 


