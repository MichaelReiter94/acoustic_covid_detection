parameters = dict(
    batch=[8],
    lr=[1e-10],
    wd=[1e-4],
    lr_decay=[0.8],
    mixup_a=[0.2],
    mixup_p=[0.8],
    use_augm_datasets=[True],
    shift=[True],
    sigma=[0.2],
    weighted_sampler=[True],
    class_weight=[1],
    bag_size=[16],
    n_MIL_Neurons=[32],
    time_steps=[336],
    lr_in=[None],
    lr_mil=[1e1],
    dropout_p=[0.25],
    focal_loss=[0],
    exclude_outliers=[0],
    min_quality=["t1val1"],
    use_resnorm=[True],
    val_oversampl=[8]
)
# if USE_MIL is True, VAL_SET_OVERSAMPLING_FACTOR will be set to be 1 (1 means no oversampling)
# VAL_SET_OVERSAMPLING_FACTOR = parameters["val_oversampl"]

DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"
RUN_COMMENT = f""
n_epochs = 50
n_cross_validation_runs = 1
USE_MIL = False
MODEL_NAME = "resnet18"

USE_TRAIN_VAL_TEST_SPLIT = True  # use a 70/15/15 split instead of an 80/20 split without test set
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 512

# LOAD_FROM_DISC = False
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-19_epoch47_AUCROC0.8215.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-22_epoch116_AUCROC0.8163335060586734.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-06-22_finalepoch_AUCROC0.7977798150510205.pth"
# LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch104_evalMetric_77.9_seed99468865.pth"
LOAD_FROM_DISC = r"data\Coswara_processed\models\2023-07-02_epoch119_evalMetric_81.9_seed215674.pth"

SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
