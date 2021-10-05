# Dataset Structural Index: Understanding a machine's perspective towards visual data

There are three parts to this code.
1. Feature extraction
2. Similarity matrix generation
3. Variety contribution ratio calculation

## Code usage

### Folder management for feature extraction
You need to add your images into a directory called __database/__, so it will look like this:

    ├── src/            # Source files
    ├── cache/          # Generated on runtime for feature extraction file
    ├── models/         # Containing all the model training files
    ├── README.md       # Intro to the repo
    └── database/       # Directory of all your images

__all your images should be put into database/__

In this directory, each image class should have its own directory and the images belonging to that class should put into that directory.

To get started with feature extraction, run the feature extraction code through ```python resnet.py``` after following the env steps and folder management as described there. 
Once you run the above code, visit the cache/ directory where you will find hte extracted features file. The same file will be used in the next step.

### Calculating the similarity matrix

After the feature extraction step is completed the Similarity matrix and variety can be generated by running the DSI class from the DSI - ```Similarity matrix and Variety contribution ratio calculation.py file``` 
Generation of Similarity matrix and variety is followed by the code for variety contribution ratio and to remove the redundant images for dataset optimization.

## Results of training on optimized dataset with same model architectures and pipeline

### Validation accuracy comparision between both versions of Stanford dogs dataset
<img src="/images/figure3-1.png" width="700">

### Validation accuracy comparision between both versions of Oxford flowers dataset
<img src="/images/figure7-1.png" width="700">

## Credits

Original dataset credits are to their respective authors:
1. A. Khosla, N. Jayadevaprakash, B. Yao, F.-F. Li, Novel dataset for fine-grained image categorization: Stanford dogs, in: Proc. CVPR Workshop
on Fine-Grained Visual Categorization (FGVC), Vol. 2, 2011.
2. Nilsback, Maria-Elena, and Andrew Zisserman. "Automated flower classification over a large number of classes." 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing. IEEE, 2008.

Feature extraction is based on the work of Po-Chih Huang's CBIR system based on ResNet features.

If you want to cite the entire work of Dataset Structural Index: Understanding a machine's perspective towards visual data please make sure to include the full citiation as follows:

## Author
Dishant Parikh | [DishantP](https://github.com/Dishant-P)
