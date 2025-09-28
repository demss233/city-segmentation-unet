# UNet + K-Means City Semantic Segmentation

A PyTorch project that combines **UNet** for pixel-wise semantic segmentation with **K-Means clustering** for post-processing, aimed at urban scene understanding.

<img width="774" height="714" alt="image" src="https://github.com/user-attachments/assets/f3111c8e-1df3-4077-9f9f-38e38d4f07f6" />

→ NOTE: You can visit my [Notebook](https://www.kaggle.com/code/shivsatyam/unet-torch-semantic-segmentation-of-city-images) for more details on outputs.

## Features
- UNet architecture for semantic segmentation
- K-Means clustering to refine segmentation boundaries
- Tailored for cityscape imagery

## Usage
1. **Create a virtual environment**
 - **On Linux/Mac:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
 - **On Windows (Command Prompt):**
  ```bash
  python -m venv venv
  venv\Scripts\activate
```

2. **Install dependencies**

Since I don’t have a `requirements.txt` yet, manually install the main packages:
```bash
pip install torch torchvision tqdm matplotlib pandas numpy
```

3. **Dataset**

  - The dataset for this implementation can be found at [This Link](https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs).
  -  After downloading the dataset, make a new folder '**data**' and paste the **dataset** there.
  -  then run python train.py and you shall see the training process.

## Losses & Plots
<img width="900" height="548" alt="image" src="https://github.com/user-attachments/assets/2b905878-72e3-4298-a1bb-999b1223fcf0" />

## Research Papers
- Ronneberger, O., Fischer, P., & Brox, T. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- MacQueen, J. (1967). [Some Methods for Classification and Analysis of Multivariate Observations](https://projecteuclid.org/euclid.bsmsp/1183143727)


