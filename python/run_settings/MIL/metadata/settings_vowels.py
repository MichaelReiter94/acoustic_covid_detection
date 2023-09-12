# ########################################  MIL MIL MIL MIL MIL  #######################################################
parameters = dict(
    batch=[12],
    lr=[3e-6],
    lr_decay=[0.2],
    lr_in=[0.2],
    wd=[1e-5],

    normalize=[False],
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
    lr_mil=[10],
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
    resnorm_affine=[True],
    resnorm_gamma=[0.85],
    input_resnorm=[True],
    track_stats=[False],
    dropout_p_MIL=[0.1]
)
# #########################################  MIL MIL MIL MIL MIL  ######################################################

USE_MIL = True

RUN_COMMENT = f"metadataMIL_excludedTypeOfTest_saveweights"
n_epochs = 1
n_cross_validation_runs = 5


SAVE_TO_DISC = False
EVALUATE_TEST_SET = True
ID_PERFORMANCE_TRACKING = "correct_metadata.pickle"
LOAD_FROM_DISC = False
# LOAD_FROM_DISC = "2023-07-06_epoch91_evalMetric_83.7_seed99468865_vowels.pth"
# LOAD_FROM_DISC_multipleSplits = [
#     "224_timesteps2023-08-15_epoch79_evalMetric_77.7_combined_vowels _seed99468865.pth",
#     "224_timesteps2023-08-15_epoch104_evalMetric_74.7_combined_vowels _seed215674.pth",
#     "224_timesteps2023-08-15_epoch83_evalMetric_76.3_combined_vowels _seed3213213211.pth",
#     "224_timesteps2023-08-15_epoch50_evalMetric_82.4_combined_vowels _seed55555555.pth",
#     "224_timesteps2023-08-15_epoch82_evalMetric_77.8_combined_vowels _seed66445511337.pth"
# ]

# LOAD_FROM_DISC_multipleSplits = [
#     "448_timesteps2023-08-15_epoch66_evalMetric_78.8_combined_vowels _seed99468865.pth",
#     "448_timesteps2023-08-15_epoch120_evalMetric_75.5_combined_vowels _seed215674.pth",
#     "448_timesteps2023-08-15_epoch31_evalMetric_77.3_combined_vowels _seed3213213211.pth",
#     "448_timesteps2023-08-15_epoch18_evalMetric_81.6_combined_vowels _seed55555555.pth",
#     "448_timesteps2023-08-15_epoch78_evalMetric_81.0_combined_vowels _seed66445511337.pth"
# ]

LOAD_FROM_DISC_multipleSplits = [
    "336_timesteps2023-09-01_epoch195_evalMetric_81.7_combined_vowels _seed99468865.pth",
    "336_timesteps2023-09-01_epoch130_evalMetric_74.9_combined_vowels _seed215674.pth",
    "336_timesteps2023-09-01_epoch37_evalMetric_77.7_combined_vowels _seed3213213211.pth",
    "336_timesteps2023-09-01_epoch187_evalMetric_81.8_combined_vowels _seed55555555.pth",
    "336_timesteps2023-09-01_epoch182_evalMetric_80.1_combined_vowels _seed66445511337.pth"
]

FREEZE_MODEL = False
# ###########################################   DO NOT CHANGE LINES BELOW   ############################################
DATASET_NAME = "2023_06_25_logmel_combined_vowels_NEW_23msHop_96msFFT_fmax11000_224logmel"
MODEL_NAME = "resnet18"
QUICK_TRAIN_FOR_TESTS = False
SAMPLES_PER_EPOCH = 1024
TRAIN_ON_FULL_SET = False
USE_TRAIN_VAL_TEST_SPLIT = True
