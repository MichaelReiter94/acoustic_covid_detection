# ########################################  MIL MIL MIL MIL MIL  #######################################################
parameters = dict(
    batch=[12],
    lr=[1e-5],
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

    bag_size=[18],
    lr_mil=[1],
    n_MIL_Neurons=[256],

    exclude_conf_miscl=[True],
    self_assessment_penalty=[0.85],
    val_oversampl=[1],
    class_weight=[1],
    weighted_sampler=[False],
    time_domain_augmentations_pos=[0],
    time_domain_augmentations_neg=[0],
    exclude_exposed=[True],

    use_resnorm=[False],
    resnorm_affine=[False],
    resnorm_gamma=[None],
    input_resnorm=[False],
    track_stats=[False],
    dropout_p_MIL=[0.10]
)
# ########################################  MIL MIL MIL MIL MIL  #######################################################

USE_MIL = True

RUN_COMMENT = f"metadataMIL_excludedTypeOfTest_saveweights"
n_epochs = 1
n_cross_validation_runs = 5


SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
ID_PERFORMANCE_TRACKING = "correct_metadata.pickle"

LOAD_FROM_DISC = False
# LOAD_FROM_DISC = "2023-07-06_epoch169_evalMetric_83.8_seed99468865_cough.pth"
# LOAD_FROM_DISC_multipleSplits = [
#     "2023-07-11_epoch78_evalMetric_82.4_combined_coughs _seed99468865.pth",
#     "2023-07-11_epoch11_evalMetric_82.6_combined_coughs _seed215674.pth",
#     "2023-07-11_epoch76_evalMetric_83.8_combined_coughs _seed3213213211.pth",
#     "2023-07-11_epoch52_evalMetric_84.8_combined_coughs _seed55555555.pth",
#     "2023-07-11_epoch48_evalMetric_84.7_combined_coughs _seed66445511337.pth"]

# LOAD_FROM_DISC_multipleSplits = [
#     "224_timesteps2023-08-15_epoch68_evalMetric_81.6_combined_coughs _seed99468865.pth",
#     "224_timesteps2023-08-15_epoch82_evalMetric_75.8_combined_coughs _seed215674.pth",
#     "224_timesteps2023-08-15_epoch117_evalMetric_78.0_combined_coughs _seed3213213211.pth",
#     "224_timesteps2023-08-15_epoch46_evalMetric_82.7_combined_coughs _seed55555555.pth",
#     "224_timesteps2023-08-15_epoch15_evalMetric_77.2_combined_coughs _seed66445511337.pth"
# ]

# LOAD_FROM_DISC_multipleSplits = [
#     "448_timesteps2023-08-15_epoch115_evalMetric_81.9_combined_coughs _seed99468865.pth",
#     "448_timesteps2023-08-15_epoch81_evalMetric_79.3_combined_coughs _seed215674.pth",
#     "448_timesteps2023-08-15_epoch40_evalMetric_80.4_combined_coughs _seed3213213211.pth",
#     "448_timesteps2023-08-15_epoch7_evalMetric_82.6_combined_coughs _seed55555555.pth",
#     "448_timesteps2023-08-15_epoch92_evalMetric_78.9_combined_coughs _seed66445511337.pth"
# ]

LOAD_FROM_DISC_multipleSplits = [
    "336_timesteps2023-09-01_epoch197_evalMetric_81.7_combined_coughs _seed99468865.pth",
    "336_timesteps2023-09-01_epoch158_evalMetric_78.6_combined_coughs _seed215674.pth",
    "336_timesteps2023-09-01_epoch181_evalMetric_81.9_combined_coughs _seed3213213211.pth",
    "336_timesteps2023-09-01_epoch93_evalMetric_82.9_combined_coughs _seed55555555.pth",
    "336_timesteps2023-09-01_epoch85_evalMetric_81.7_combined_coughs _seed66445511337.pth"
]
FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_05_22_logmel_combined_coughs_NEW_11msHop_23msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
