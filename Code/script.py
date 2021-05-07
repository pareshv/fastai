from fastai.vision.all import * 
from fastai.vision.widgets import *

path = untar_data('https://github.com/pareshv/fastai/raw/main/mousedetection_05042021_2.zip', force_download=True, dest='drive/My Drive/data')


data =  DataBlock(blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42), 
        get_y=parent_label,
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

dls = data.dataloaders(path, bs=10)

dls.show_batch(nrows=1, ncols=5)

learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(5, nrows=1, figsize = (25,5))

learn.recorder.plot_loss()

learn.export('hacker_vs_genuine_05072021.pkl')

