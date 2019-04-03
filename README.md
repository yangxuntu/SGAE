# SGAE/ pytorch 0.3.0
Auto-Encoding Scene Graphs for Image Captioning, CVPR 2019

# Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.

And we use the visual features provided by paper Bottom-up and top-down attention for image captioning and visual question answering in https://github.com/peteanderson80/bottom-up-attention.

If you like this code, please consider to cite their corresponding papers and my CVPR paper.

# Installation anaconda and the environment
I provide the anaconda environment for running my code in https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing. You should download the file ``environment_yx1.yml'' from this link and set up the environment as follows.
1. Download the anaconda from the website https://www.anaconda.com/ and install it.
2. Go to website https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment to learn how to learn how to 'creating an environment from an environment.yml file'.
```
conda env create -f environment_yx1.yml
```
3. After installing anaconda and setting up the environment, run the following code to get into the environment.
```
source activate yx1
```
If you want to exit from this environment, you can run the following code to exit.
```
source deactivate
```

# Downloading meta data, e.g., image captions and visual features.
1.
