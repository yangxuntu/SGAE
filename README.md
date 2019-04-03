# SGAE/ pytorch 0.3.0
Auto-Encoding Scene Graphs for Image Captioning, CVPR 2019

# Acknowledgement
This code is implemented based on Ruotian Luo's implementation of image captioning in https://github.com/ruotianluo/self-critical.pytorch.

And we use the visual features provided by paper Bottom-up and top-down attention for image captioning and visual question answering in https://github.com/peteanderson80/bottom-up-attention.

If you like this code, please consider to cite their corresponding papers and my CVPR paper.

# Installation anaconda and the environment
I provide the anaconda environment for running my code in https://drive.google.com/drive/folders/1GvwpchUnfqUjvlpWTYbmEvhvkJTIWWRb?usp=sharing. You should download the file ''environment_yx1.yml'' from this link and set up the environment as follows.
1.Download the anaconda from the website https://www.anaconda.com/ and install it.
2.Go to website https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html?highlight=environment to learn how to learn how to 'creating an environment from an environment.yml file'.
```
conda env create -f environment_yx1.yml
```
3.After installing anaconda and setting up the environment, run the following code to get into the environment.
```
source activate yx1
```
If you want to exit from this environment, you can run the following code to exit.
```
source deactivate
```

# Downloading meta data, e.g., image captions, visual features, image scene graphs, sentence scene graphs.
You can get more details from  https://github.com/ruotianluo/self-critical.pytorch.

1.Download preprocessed coco captions from link from Karpathy's homepage. Extract dataset_coco.json from the zip file and copy it in to data/. This file provides preprocessed captions and also standard train-val-test splits.
The do:
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
prepro_labels.py will map all words that occur <= 5 times to a special UNK token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into data/cocotalk.json and discretized caption data are dumped into data/cocotalk_label.h5.

2.Download Bottom-up features.
Download pre-extracted feature from https://github.com/peteanderson80/bottom-up-attention. You can either download adaptive one or fixed one. We use the ''10 to 100 features per image (adaptive)'' in our experiments.
For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```
Then :
```
python script/make_bu_data.py --output_dir data/cocobu
```
This will create data/cocobu_fc, data/cocobu_att and data/cocobu_box. If you want to use bottom-up feature, you can just follow the following steps and replace all cocotalk with cocobu.
