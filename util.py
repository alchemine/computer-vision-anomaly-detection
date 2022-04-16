from analysis_tools.common import *
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.metrics import F1Score


def load_data(size=512):
    @delayed
    def load_img(path):
        return cv2.resize(cv2.imread(path), (size, size))

    train_full_data_meta = pd.read_csv(join(PATH.input, 'train_df.csv'), index_col=0)
    test_data_meta       = pd.read_csv(join(PATH.input, 'test_df.csv'), index_col=0)

    with ProgressBar():
        X_train_full_class = np.array(compute(*[load_img(join(PATH.train, name)) for name in train_full_data_meta['file_name']]))
        X_test             = np.array(compute(*[load_img(join(PATH.test, name))  for name in test_data_meta['file_name']]))

    return train_full_data_meta, test_data_meta, X_train_full_class, X_test


def preprocess(ds, training, batch_size, augment=True):
    aug_model = keras.models.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    ds = ds.cache().batch(batch_size)
    if training:
        ds = ds.shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(lambda X, y, sw: (aug_model(X), y, sw), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def build_model(n_classes, strategy, input_shape):
    def fine_tuning_on():
        with strategy.scope():
            base_model.trainable = False
            model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=[F1Score(num_classes=n_classes, average='macro')])

    with strategy.scope():
        base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape)
        base_model.trainable = False
        inputs = keras.Input(input_shape)
        hidden = base_model(inputs, training=False)
        hidden = keras.layers.GlobalAveragePooling2D()(hidden)
        outputs = keras.layers.Dense(n_classes, activation='softmax')(hidden)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=[F1Score(num_classes=n_classes, average='macro')])

    return model, fine_tuning_on
