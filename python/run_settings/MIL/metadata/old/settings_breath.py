# ########################################  MIL MIL MIL MIL MIL  #######################################################
parameters = dict(
    batch=[6],
    lr=[3e-6],
    lr_decay=[0.2],
    lr_in=[0.2],
    wd=[1e-5],

    normalize=[True],
    time_steps=[336],
    use_augm_datasets=[False],
    shift=[False],
    sigma=[0],
    mixup_a=[0],
    mixup_p=[0],
    dropout_p=[0],
    transfer_func_sim=[0],
    random_gain=[0],
    exclude_outliers=[0],
    focal_loss=[0],

    bag_size=[8],
    lr_mil=[0.25],
    n_MIL_Neurons=[64],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[1.0],
    val_oversampl=[1],
    class_weight=[1],
    weighted_sampler=[False],
    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[0],
    exclude_exposed=[True],

    use_resnorm=[True],
    resnorm_affine=[True],
    resnorm_gamma=[0.85],
    input_resnorm=[True],
    track_stats=[False],
    dropout_p_MIL=[0.0]
)
# ########################################  MIL MIL MIL MIL MIL  #######################################################

USE_MIL = True

RUN_COMMENT = f"metadataMIL_saveIDPerformance"
n_epochs = 1
n_cross_validation_runs = 5

SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
ID_PERFORMANCE_TRACKING = "metadata_mil.pickle"

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = "2023-06-12_resnet18_23ms_82_7_AUCROC_iter74.pth"
# LOAD_FROM_DISC = "2023-07-11_epoch102_evalMetric_81.9_combined_breaths _seed99468865.pth"

# LOAD_FROM_DISC_multipleSplits = [
#     "224_timesteps2023-08-15_epoch99_evalMetric_74.5_combined_breaths _seed99468865.pth",
#     "224_timesteps2023-08-15_epoch87_evalMetric_79.5_combined_breaths _seed215674.pth",
#     "224_timesteps2023-08-15_epoch119_evalMetric_76.8_combined_breaths _seed3213213211.pth",
#     "224_timesteps2023-08-15_epoch46_evalMetric_79.1_combined_breaths _seed55555555.pth",
#     "224_timesteps2023-08-15_epoch56_evalMetric_76.4_combined_breaths _seed66445511337.pth"
# ]
# LOAD_FROM_DISC_multipleSplits = [
#     "448_timesteps2023-08-15_epoch108_evalMetric_77.5_combined_breaths _seed99468865.pth",
#     "448_timesteps2023-08-15_epoch61_evalMetric_80.0_combined_breaths _seed215674.pth",
#     "448_timesteps2023-08-15_epoch117_evalMetric_76.8_combined_breaths _seed3213213211.pth",
#     "448_timesteps2023-08-15_epoch108_evalMetric_79.4_combined_breaths _seed55555555.pth",
#     "448_timesteps2023-08-15_epoch39_evalMetric_77.1_combined_breaths _seed66445511337.pth"
# ]

LOAD_FROM_DISC_multipleSplits = [
    "336_timesteps2023-09-01_epoch196_evalMetric_77.1_combined_breaths _seed99468865.pth",
    "336_timesteps2023-09-01_epoch161_evalMetric_81.7_combined_breaths _seed215674.pth",
    "336_timesteps2023-09-01_epoch115_evalMetric_77.5_combined_breaths _seed3213213211.pth",
    "336_timesteps2023-09-01_epoch189_evalMetric_79.5_combined_breaths _seed55555555.pth",
    "336_timesteps2023-09-01_epoch112_evalMetric_76.7_combined_breaths _seed66445511337.pth"
]

FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "logmel_combined_breaths_NEW_46msHop_92msFFT_fmax11000_224logmel"
# DATASET_NAME = "logmel_combined_breaths_NEW_23msHop_46msFFT_fmax11000_224logmel"

MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
