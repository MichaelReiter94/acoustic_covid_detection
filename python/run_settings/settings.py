import os
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


# MIL settings
parameters = dict(
    batch=[12],
    lr=[1e-7],
    wd=[1e-4],
    lr_decay=[0.2],
    mixup_a=[0.4],
    mixup_p=[1],
    use_augm_datasets=[False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[12],
    n_MIL_Neurons=[64],
    time_steps=[336],
    lr_in=[None],
    lr_mil=[1e4],
    dropout_p=[0.25],
    focal_loss=[0],
    exclude_outliers=[0],
    min_quality=["t1val1"],
    use_resnorm=[True],
    val_oversampl=[8]
)

# # evaluation settings for saved models
# parameters = dict(
#     batch=[64],
#     lr=[1e-10],
#     wd=[1e-4],
#     lr_decay=[0.8],
#     mixup_a=[0.2],
#     mixup_p=[0.8],
#     use_augm_datasets=[False],
#     shift=[True],
#     sigma=[0.2],
#     weighted_sampler=[True],
#     class_weight=[1],
#     bag_size=[12],
#     n_MIL_Neurons=[32],
#     time_steps=[336],
#     lr_in=[None],
#     lr_mil=[1e3],
#     dropout_p=[0.25],
#     focal_loss=[0],
#     exclude_outliers=[0],
#     min_quality=["t1val1"],
#     use_resnorm=[False],
#     val_oversampl=[8]
# )
# if USE_MIL is True, VAL_SET_OVERSAMPLING_FACTOR will be set to be 1 (1 means no oversampling)
# VAL_SET_OVERSAMPLING_FACTOR = parameters["val_oversampl"]

# DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
# DATASET_NAME = "2023_06_25_logmel_combined_speech_NEW_23msHop_46msFFT_fmax11000_224logmel"
DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
# DATASET_NAME = "2023_05_22_logmel_combined_coughs_NEW_11msHop_23msFFT_fmax11000_224logmel"


RUN_COMMENT = f""
n_epochs = 4
n_cross_validation_runs = 1
USE_MIL = True
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 256

# LOAD_FROM_DISC = False
LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch104_evalMetric_77.9_seed99468865.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch119_evalMetric_81.9_seed215674.pth"

# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch83_evalMetric_85.7_seed99468865_speech.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-09_epoch149_evalMetric_84.9_seed99468865_vowels.pth"



if isinstance(LOAD_FROM_DISC, str):
    LOAD_FROM_DISC = os.path.join(*LOAD_FROM_DISC.split("\\"))
SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
