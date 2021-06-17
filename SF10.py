from fastai.vision import *

from google.colab import drive
drive.mount('/content/gdrive')

path = Path('/content/gdrive/MyDrive/Data2') # Define path to the image folders

np.random.seed(42)

data_squished = ImageDataBunch.from_folder(path, 
                                  train=".", 
                                  valid_pct=0.9,
                                  ds_tfms=get_transforms(), 
                                 size=(450,450),#instead of size=450
                                  num_workers=4, 
                                  bs = 16) \
        .normalize(imagenet_stats)

data_squished.c

len(data_squished.train_ds)

len(data_squished.valid_ds)

learn2 = cnn_learner(data_squished, 
                     models.resnet50,       
                     metrics=[error_rate, accuracy])

learn2.fit_one_cycle(1)

learn2.unfreeze()
learn2.lr_find()
learn2.recorder.plot()

learn2.fit_one_cycle(3, max_lr=slice(1e-5,1e-4))

interp = ClassificationInterpretation.from_learner(learn2)
interp.plot_confusion_matrix()

