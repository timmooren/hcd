# Log Tim Mooren
Student number: 11710160


Monday 14 June:
- Creating GitHub repository
- Creating new Colab Notebook
- Downloading Kaggle data to Colab
- Recovering personal Drive data accidentally deleted from Colab

Tuesday 15 June:
- Still recovering Drive data
- Image data augmentations that I expect may work:
    - Zoom
    - Tilt
    - Shift
    - Mirroring
- Image data augmentations that I expect may be counterproductive:
    - Brightness
    - Stretch

- Physical meetup with group:
    - For pair programming, I functioned as 'pilot'
    - Write documentation and create sections in colab
    - Write in report file (section 2, 3 and 4)

Wednesday 16 June:
- Ran dropout model with more epochs
- Written and ran two new models (deeper and experimental)
- Recovering Drive data

Thursday 17 June:
(Meeting with group and Wouter)
- Re-ran some models with more epochs
- Written and ran batch normalization and simple data augmentation model
- Added descriptions to new models and some of the old ones.
- Added analyses to the model graphs of yesterday and today.
- Looked into k-folding for CNNs
- Searched for other metrics that may be more applicable than accuracy (e.g. precision or recall)
- Tried to implement mini-batch using ADAM optimizer, but failed


Friday 18 June:
- Running data augmentation network with higher dropout to see if this can solve the validation fluctuations
- Running data augmentation network with larger batch size to see if this can solve the validation fluctuations
- Running deep network with different random state to see whether the test/val distribution may cause the validation fluctuations
- Researched and written on other metrics (precision, recall, sensitivity, specificity)

Sunday 20 June
- Reread the data description and discovered that the images are 96x96, while we are using a 32x32 input on the CNN.
- Adjusted the base model to 96x96 and ran it again, but did not have enough RAM.
