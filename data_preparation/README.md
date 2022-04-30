`img_combine.py` combines all the images from different folder.
`data_preparation.py`  extracts traditional features from each image and outputs an excel table that contains the images.
`img_tag.py` uses PyGame to classify a small part of images. The images are divided into three classes: HR,LR and other(not emoji).
`img_classify.py` construct a decision tree classifier and save the model using `joblib`.
`make_dataset.py` use the trained decision tree classifier to classifier our dataset.
`make_lr.py` use high resolution image to generate low resolution image.
