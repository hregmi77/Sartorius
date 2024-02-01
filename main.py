## Helpful Links ##
# https://www.kaggle.com/ishandutta/sartorius-indepth-eda-explanation-model/notebook
# https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-keras-u-net-training
import warnings
warnings.filterwarnings("ignore")
import os
import json
import cv2
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import segmentation_models as sm
## Main Segmentation Code
train_df = pd.read_csv('input/sartorius-cell-instance-segmentation/train.csv')
print(train_df.shape)
train_df.head(4)
train_df=train_df.head(n=10000)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)
def build_masks(image_id,input_shape, colors=True):
    height, width = input_shape
    labels = train_df[train_df["id"] == image_id]["annotation"].tolist()
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask

# sample_filename = '0030fd0e6378'
# sample_image_df = train_df[train_df['id'] == sample_filename]
# sample_path = f"input/sartorius-cell-instance-segmentation/train/{sample_image_df['id'].iloc[0]}.png"
# sample_img = cv2.imread(sample_path)
# sample_rles = sample_image_df['annotation'].values
#
# sample_masks1=build_masks(sample_filename,input_shape=(520, 704), colors=False)
# sample_masks2=build_masks(sample_filename,input_shape=(520, 704), colors=True)
#
# fig, axs = plt.subplots(3, figsize=(20, 20))
# axs[0].imshow(sample_img)
# axs[0].axis('off')
#
# axs[1].imshow(sample_masks1)
# axs[1].axis('off')
#
# axs[2].imshow(sample_masks2)
# axs[2].axis('off')
# plt.show()


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='input/sartorius-cell-instance-segmentation/train',
                 batch_size=32, dim=(256, 256), n_channels=3,
                 n_classes=3, random_state=2019, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y

        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['id'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}.png"
            img = self.__load_grayscale(img_path)

            # Store samples
            X[i,] = img

        return X

    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['id'].iloc[ID]
            image_df = self.target_df[self.target_df['id'] == im_name]

            # rles = image_df['annotation'].values
            masks = build_masks(im_name, self.dim, colors=False)

            y[i,] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # resize image
        dsize = (256, 256)
        img = cv2.resize(img, dsize)

        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img
BATCH_SIZE = 4

train_idx, val_idx = train_test_split(
    train_df.index, random_state=2019, test_size=0.2 # mask_count_df
)

train_generator = DataGenerator(
    train_idx,
    df=train_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    n_classes=3
)

val_generator = DataGenerator(
    val_idx,
    df=train_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    n_classes=3
)
# for i in range(1):
#     images, label = val_generator[i]
#     print("Dimension of the CT scan is:", images.shape)
#     print("label is:", label.shape)
#     plt.imshow(images[0,:,:,0], cmap="gray")
#     plt.show()
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)

sm.set_framework('tf.keras')
sm.framework()

from segmentation_models import Unet
from segmentation_models.utils import set_trainable


model = Unet('efficientnetb0',input_shape=(256, 256, 3), classes=3, activation='sigmoid',encoder_weights='imagenet')

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=300,
)
#inp = Input(shape=(512, 640, 1))
#l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
#out = base_model(l1)
#model = Model(inp, out, name=base_model.name)

model.compile(optimizer='adam', loss=bce_dice_loss,metrics=[dice_coef,iou_coef,'accuracy']) #bce_dice_loss binary_crossentropy
model.summary()


checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_dice_coef',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    callbacks=[checkpoint],
    use_multiprocessing=False,
    workers=4,
    epochs=100
)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('history.csv')

# PLOT TRAINING
plt.figure(figsize=(15,5))
plt.plot(range(history.epoch[-1]+1),history.history['val_iou_coef'],label='Val_iou_coef')
plt.plot(range(history.epoch[-1]+1),history.history['iou_coef'],label='Trn_iou_coef')
plt.title('IOU'); plt.xlabel('Epoch'); plt.ylabel('iou_coef');plt.legend();
plt.show()

# PLOT TRAINING
plt.figure(figsize=(15,5))
plt.plot(range(history.epoch[-1]+1),history.history['val_dice_coef'],label='Val_dice_coef')
plt.plot(range(history.epoch[-1]+1),history.history['dice_coef'],label='Trn_dice_coef')
plt.title('DICE'); plt.xlabel('Epoch'); plt.ylabel('dice_coef');plt.legend();
plt.show()

def build_rles(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
sub_df = pd.read_csv('input/sartorius-cell-instance-segmentation/sample_submission.csv')
sub_df.head()

test_imgs=sub_df

model.load_weights('model.h5')
test_df = []

for i in range(0, test_imgs.shape[0], 300):
    batch_idx = list(
        range(i, min(test_imgs.shape[0], i + 300))
    )

    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        base_path='input/sartorius-cell-instance-segmentation/test',
        target_df=sub_df,
        batch_size=1,
        n_classes=3
    )

    batch_pred_masks = model.predict(
        test_generator,
        workers=1,
        verbose=1,
        use_multiprocessing=False
    )

    for j, b in tqdm(enumerate(batch_idx)):
        filename = test_imgs['id'].iloc[b]
        image_df = sub_df[sub_df['id'] == filename].copy()

        pred_masks = batch_pred_masks[j,].round().astype(int)
        plt.imshow(pred_masks)
        plt.draw()
        plt.pause(10)

        pred_rles = build_rles(pred_masks)

        image_df['predicted'] = pred_rles
        test_df.append(image_df)
sub = pd.concat(test_df)
sub.head()
sub.to_csv('submission.csv', index=False)
