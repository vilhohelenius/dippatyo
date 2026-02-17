###############################################################
# CONFIGURATION FILE — Master Thesis ML Pipeline
# Central place for all global settings and paths
###############################################################

############################
# Reproducibility
############################

SEED <- 42
set.seed(SEED)

############################
# Paths
############################

# Root folders
DATA_DIR    <- "data"
EXPORT_DIR  <- "export"
RESULTS_DIR <- "results"
PLOTS_DIR   <- file.path(RESULTS_DIR, "plots")
MODELS_DIR  <- file.path(RESULTS_DIR, "models")

# Create folders if missing
dir.create(DATA_DIR,    showWarnings = FALSE)
dir.create(EXPORT_DIR,  showWarnings = FALSE)
dir.create(RESULTS_DIR, showWarnings = FALSE)
dir.create(PLOTS_DIR,   showWarnings = FALSE)
dir.create(MODELS_DIR,  showWarnings = FALSE)

############################
# Dataset split
############################

TRAIN_FRACTION <- 0.80

############################
# Scaling experiment sizes
############################
# Used for scaling experiment

DATASET_SIZES <- c(
  50,
  100,
  500,
  1000,
  2000,
  5000,
  8000   # last value automatically clipped to train size if needed
)

############################
# Evaluation metrics
############################

PRIMARY_METRIC <- "RMSE"
ALL_METRICS <- c("R2", "RMSE", "MAE")

############################
# Cross-validation
############################

CV_FOLDS <- 5
EARLY_STOPPING_ROUNDS <- 30
MAX_ROUNDS <- 1000

############################
# XGBoost default parameters
############################

XGB_PARAMS <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

############################
# CatBoost default parameters
############################

CATBOOST_PARAMS <- list(
  loss_function = "RMSE",
  iterations = 1000,
  depth = 6,
  learning_rate = 0.05,
  random_seed = SEED,
  od_type = "Iter",
  od_wait = 30,
  verbose = FALSE
)

############################
# Lasso (glmnet)
############################

LASSO_ALPHA <- 1  # alpha = 1 → Lasso

############################
# SHAP settings
############################

SHAP_TOP_FEATURES <- 20

############################
# Metadata column names
############################

TARGET_COL <- "bmi"
AGE_COL    <- "age_years"
SEX_COL    <- "sex"

############################
# TabPFN export filenames
############################

TRAIN_FULL_FILE <- file.path(EXPORT_DIR, "train_full.csv")
TEST_FULL_FILE  <- file.path(EXPORT_DIR, "test_full.csv")

TRAIN_MICRO_FILE <- file.path(EXPORT_DIR, "train_microbiome.csv")
TEST_MICRO_FILE  <- file.path(EXPORT_DIR, "test_microbiome.csv")

TRAIN_META_FILE <- file.path(EXPORT_DIR, "train_metadata.csv")
TEST_META_FILE  <- file.path(EXPORT_DIR, "test_metadata.csv")

############################
# Helper: safe dataset sizes
############################

clip_dataset_sizes <- function(train_n) {
  unique(pmin(DATASET_SIZES, train_n))
}