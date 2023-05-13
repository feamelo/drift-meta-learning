SEED = 2024
BASE_CONCEPT_SIZE = 10000
NEW_CONCEPT_SIZE = 5000
RECURRING_CONCEPT_SIZE = 1000

DATASETS_METADATA = [
    # Sine
    {
        "dataset_name": "Sine",
        "file_name": "abrupt_sine_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 3, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True, "has_noise": False}}]
    },
    {
        "dataset_name": "Sine",
        "file_name": "abrupt_sine_unbalanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": False, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": False, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 3, "seed": SEED, "balance_classes": False, "has_noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": False, "has_noise": False}}]
    },
    {
        "dataset_name": "Sine",
        "file_name": "abrupt_sine_balanced_noise",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": True, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 3, "seed": SEED, "balance_classes": True, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True, "has_noise": True}}]
    },
    {
        "dataset_name": "Sine",
        "file_name": "abrupt_sine_unbalanced_noise",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": False, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": False, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 3, "seed": SEED, "balance_classes": False, "has_noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": False, "has_noise": True}}]
    },
    {
        "dataset_name": "Sine",
        "file_name": "abrupt_recurring_sine_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 3, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True, "has_noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True, "has_noise": False}}]
    },

    # SEA
    {
        "dataset_name": "SEA",
        "file_name": "abrupt_sea",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 1, "seed": SEED, "noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 2, "seed": SEED, "noise": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 3, "seed": SEED, "noise": False}}]
    },
    {
        "dataset_name": "SEA",
        "file_name": "abrupt_sea_noise",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 1, "seed": SEED, "noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 2, "seed": SEED, "noise": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"variant": 3, "seed": SEED, "noise": True}}]
    },
    {
        "dataset_name": "SEA",
        "file_name": "abrupt_recurring_sea",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 1, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 2, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 3, "seed": SEED, "noise": False}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"variant": 0, "seed": SEED, "noise": False}}]
    },

    # STAGGER
    {
        "dataset_name": "STAGGER",
        "file_name": "abrupt_stagger_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": True}}]
    },
    {
        "dataset_name": "STAGGER",
        "file_name": "abrupt_stagger_unbalanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": False}},]
    },
    {
        "dataset_name": "STAGGER",
        "file_name": "abrupt_recurring_stagger_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 2, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}}]
    },

    # Mixed
    {
        "dataset_name": "Mixed",
        "file_name": "abrupt_mixed_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True}}]
    },
    {
        "dataset_name": "Mixed",
        "file_name": "abrupt_mixed_unbalanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": False}},
            {"n_inst": NEW_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": False}}]
    },
    {
        "dataset_name": "Mixed",
        "file_name": "abrupt_recurring_mixed_balanced",
        "params": [
            {"n_inst": BASE_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 0, "seed": SEED, "balance_classes": True}},
            {"n_inst": RECURRING_CONCEPT_SIZE, "config": {"classification_function": 1, "seed": SEED, "balance_classes": True}}]
    },
]
