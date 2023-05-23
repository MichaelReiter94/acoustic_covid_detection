parameters = dict(
    batch=[64],
    lr=[5e-4, 1e-4],
    wd=[1e-4],
    lr_decay=[0.99],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[True, False],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[8],
    n_MIL_Neurons=[64],
    time_steps=[200],
    lr_in=[None],
    dropout_p=[0.0, 0.15, 0.3],
    focal_loss=[0],
    min_quality=[1, 2]
)
DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
RUN_COMMENT = f"addedAugments_dropoutP"
n_epochs = 250
n_cross_validation_runs = 1
USE_MIL = False
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
