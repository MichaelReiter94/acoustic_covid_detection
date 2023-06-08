parameters = dict(
    batch=[64],
    lr=[1e-3, 5e-4, 1e-4],
    wd=[1e-4],
    lr_decay=[0.99],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[True, False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[10],
    n_MIL_Neurons=[64],
    time_steps=[200, 300],
    lr_in=[None],
    lr_mil=[1.0],
    dropout_p=[0.1, 0.3],
    focal_loss=[0],
    exclude_outliers=[0],
    min_quality=["t1val1"],
    use_resnorm=[True, False]
)
DATASET_NAME = "logmel_combined_breaths_NEW_11msHop_46msFFT_fmax11000_224logmel"
RUN_COMMENT = f"resnormTests"
n_epochs = 250
n_cross_validation_runs = 1
USE_MIL = False
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
