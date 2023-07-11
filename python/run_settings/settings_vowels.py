parameters = dict(
    batch=[64],
    lr=[1e-3, 5e-4, 1e-4, 7e-5],
    wd=[1e-4],
    lr_decay=[0.99, 0.975],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[True, False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[16],
    n_MIL_Neurons=[32],
    time_steps=[224, 336],
    lr_in=[None],
    lr_mil=[1e1],
    dropout_p=[0.1, 0.25],
    focal_loss=[0],
    exclude_outliers=[0],
    min_quality=["t1val1"],
    use_resnorm=[False],
    val_oversampl=[6]
)
# if USE_MIL is True, VAL_SET_OVERSAMPLING_FACTOR will be set to be 1 (1 means no oversampling)
# VAL_SET_OVERSAMPLING_FACTOR = parameters["val_oversampl"]

DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
RUN_COMMENT = f"hyperparams"
n_epochs = 100
n_cross_validation_runs = 1
USE_MIL = False
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024

LOAD_FROM_DISC = False
SAVE_TO_DISC = True
EVALUATE_TEST_SET = False
