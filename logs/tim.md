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
Further reading: https://librepathology.org/wiki/Lymph_node_metastasis

- Physical meetup with group:
    - For pair programming, I functioned as 'pilot/driver'
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
- Looked into k-folding for CNNs (https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/)
- Searched for other metrics that may be more applicable than accuracy (e.g. precision or recall)
- Tried to implement mini-batch using ADAM optimizer, but failed. To try next time


Friday 18 June:
- Running data augmentation network with higher dropout to see if this can solve the validation fluctuations
- Running data augmentation network with larger batch size to see if this can solve the validation fluctuations (https://towardsdatascience.com/the-3-best-optimization-methods-in-neural-networks-40879c887873, https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338)
- Running deep network with different random state to see whether the test/val distribution may cause the validation fluctuations
- Researched and written on other metrics (precision, recall, sensitivity, specificity) (https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall, https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/sensitivity-vs-specificity-statistics/)

Sunday 20 June:
- Reread the data description and discovered that the images are 96x96, while we are using a 32x32 input on the CNN
- Adjusted the base model to 96x96 and ran it again, but did not have enough RAM

Monday 22 June:
- Rewritten load_data() in the hope that this approach takes up less memory in order to prevent Colab from crashing (https://stackoverflow.com/questions/11784329/python-memory-usage-of-numpy-arrays)
- Purchased Colab Pro, adjestued and re-ran networks to 48x48 input (rather than 32x32)
- Adding more markdown comments and structure to the notebook
- Written on chapter 2, 3 and 4
- Written and ran batch size (this time it worked) and L2 regularization models
- Written crop image function so we can focus on the region of interest for the next milestone
- Interesting things for next iteration: lower learning rate, focussing on region of interest (https://towardsdatascience.com/understanding-region-of-interest-part-1-roi-pooling-e4f5dd65bb44)

