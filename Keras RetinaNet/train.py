from label_encoder import LabelEncoder
from utils import get_backbone
from loss import RetinaNetLoss
from retinanet import RetinaNet
import tensorflow as tf
import os
from dataset import data

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
#batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates)


def make_model():
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)
    return model


#callbacks_list = [
#   tf.keras.callbacks.ModelCheckpoint(
#       filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
#       monitor="loss",
#       save_best_only=False,
#       save_weights_only=True,
#       verbose=1,
#   )
#]

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
        )
    ]


# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch
train_dataset, val_dataset, _ = data()
epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset
model = make_model()

model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

