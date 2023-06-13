# parameters = dict(
#     # rand=random_seeds[:n_cross_validation_runs],
#     batch=[64],
#     lr=[7e-4],  # lr of the output layer - the lr between in/output layer are linearly interpolated
#     wd=[1e-4],  # weight decay regularization
#     lr_decay=[0.99],
#     mixup_a=[0.2],  # alpha value to decide probability distribution of how much of each of the samples is used
#     mixup_p=[0.8],  # probability of mix up being used at all
#     use_augm_datasets=[True],
#     shift=[True],
#     sigma=[0.1],
#     weighted_sampler=[True],  # whether to use a weighted random sampler to address the class imbalance
#     class_weight=[1],  # factor for loss of the positive class to address class imbalance
#     bag_size=[6],
#     n_MIL_Neurons=[64],
#     time_steps=[120],
#     lr_in=[None],  # lr of the input layer - the lr between in/output layer are linearly interpolated
#     dropout_p=[0.1],
#     focal_loss=[0],
#     # if focal_loss (gamma) == 0 it is the same as the BCE, increasing it makes it focus on harder examples.
#     # If you go below,it learns more from well classified examples and ignores more badly classified ones
#     min_quality=[1]
#     # audio quality is divided into 3 classes "0" being ba audio, "1" being medium and "2" premium quality.
#     # quality "0" is usually already removed when creating the feature set to save memory
# )
#
# # logmel_combined_breaths_NEW_11msHop_46msFFT_fmax11000_224logmel
# # logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel
# # logmel_combined_breaths_NEW_92msHop_184msFFT_fmax11000_224logmel


parameters = dict(
    batch=[64],
    lr=[1e-10],
    wd=[1e-4],
    lr_decay=[0.99],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[16],
    n_MIL_Neurons=[32],
    time_steps=[300],
    lr_in=[None],
    lr_mil=[1e2],
    dropout_p=[0.3],
    focal_loss=[0],
    exclude_outliers=[0],
    min_quality=["t1val1"],
    use_resnorm=[True]
)
DATASET_NAME = "logmel_combined_breaths_NEW_11msHop_46msFFT_fmax11000_224logmel"
RUN_COMMENT = f"pretrained_11ms"
n_epochs = 100
n_cross_validation_runs = 1
USE_MIL = False
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 8192

RANDOM_CYCLIC_SHIFT_FOR_EVALUATION = True
VAL_SET_OVERSAMPLING_FACTOR = 2
