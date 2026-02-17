###############################################################
# split_export.R
# Create reproducible train/test split and export datasets
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)

###############################################################
# Load preprocessed dataset (RDS cache)
###############################################################
df_ml <- readRDS(file.path(DATA_DIR, "df_ml_full.rds"))

set.seed(SEED)

###############################################################
# Shuffle dataset (important for fair scaling experiment)
###############################################################
df_ml <- df_ml %>% slice_sample(prop = 1)

###############################################################
# Train / Test split (80 / 20)
###############################################################
n <- nrow(df_ml)
train_size <- floor(TRAIN_FRACTION * n)

train_df <- df_ml[1:train_size, ]
test_df  <- df_ml[(train_size + 1):n, ]

cat("Train samples:", nrow(train_df), "\n")
cat("Test samples :", nrow(test_df), "\n")

###############################################################
# Build three datasets (FULL / METADATA / MICROBIOME)
###############################################################

# FULL: microbiome + age_years + sex + bmi
train_full <- train_df
test_full  <- test_df

# METADATA: age_years + sex + bmi
train_meta <- train_df %>% select(age_years, sex, bmi)
test_meta  <- test_df  %>% select(age_years, sex, bmi)

# MICROBIOME: microbiome features + bmi (exclude age_years + sex)
train_micro <- train_df %>% select(-age_years, -sex)
test_micro  <- test_df  %>% select(-age_years, -sex)

###############################################################
# Export CSV (source-of-truth for R + Python)
###############################################################
dir.create(EXPORT_DIR, showWarnings = FALSE, recursive = TRUE)

write_csv(train_full,  file.path(EXPORT_DIR, "train_full.csv"))
write_csv(test_full,   file.path(EXPORT_DIR, "test_full.csv"))

write_csv(train_meta,  file.path(EXPORT_DIR, "train_metadata.csv"))
write_csv(test_meta,   file.path(EXPORT_DIR, "test_metadata.csv"))

write_csv(train_micro, file.path(EXPORT_DIR, "train_microbiome.csv"))
write_csv(test_micro,  file.path(EXPORT_DIR, "test_microbiome.csv"))

###############################################################
# Export X/y separately
###############################################################
X_train <- train_full %>% select(-bmi)
X_test  <- test_full  %>% select(-bmi)
y_train <- train_full$bmi
y_test  <- test_full$bmi

write_csv(X_train, file.path(EXPORT_DIR, "X_train.csv"))
write_csv(X_test,  file.path(EXPORT_DIR, "X_test.csv"))
write_csv(tibble(bmi = y_train), file.path(EXPORT_DIR, "y_train.csv"))
write_csv(tibble(bmi = y_test),  file.path(EXPORT_DIR, "y_test.csv"))

###############################################################
# Export feature names (needed for SHAP / Python)
###############################################################
write_csv(
  tibble(feature = colnames(X_train)),
  file.path(EXPORT_DIR, "feature_names_full.csv")
)

X_train_micro <- train_micro %>% select(-bmi)
write_csv(
  tibble(feature = colnames(X_train_micro)),
  file.path(EXPORT_DIR, "feature_names_microbiome.csv")
)

###############################################################
# Summary
###############################################################
cat("\nSaved CSV datasets to export/:\n")
cat(" - train_full.csv / test_full.csv\n")
cat(" - train_metadata.csv / test_metadata.csv\n")
cat(" - train_microbiome.csv / test_microbiome.csv\n")
cat(" - X_train.csv / X_test.csv / y_train.csv / y_test.csv\n")
cat(" - feature_names_full.csv / feature_names_microbiome.csv\n")
cat("\nAll models (R + Python) should read from export/*.csv\n")