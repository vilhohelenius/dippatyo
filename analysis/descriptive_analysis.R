###############################################################
# analysis/descriptive_analysis.R
# Descriptive analysis: metadata + BMI + simple microbiome associations
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(ggplot2)
library(tidyr)
library(stringr)
library(tibble)
library(purrr)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# 1) Load data
###############################################################

# --- Try to load md (metadata) from RDS (recommended) ---
MD_PATH <- file.path(DATA_DIR, "md.rds")   # change if your md is saved elsewhere

if (file.exists(MD_PATH)) {
  md <- readRDS(MD_PATH)
  message("Loaded md from: ", MD_PATH)
} else {
  warning("md.rds not found at ", MD_PATH, "\nIf you don't have md saved, this script will run limited plots.")
  md <- NULL
}

# --- Load the ML dataset (microbiome-only, with bmi) from export CSVs ---
train_path <- file.path(EXPORT_DIR, "train_microbiome.csv")
test_path  <- file.path(EXPORT_DIR, "test_microbiome.csv")

if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("Missing export files:\n- ", train_path, "\n- ", test_path)
}

train_df <- read_csv(train_path, show_col_types = FALSE)
test_df  <- read_csv(test_path,  show_col_types = FALSE)

df_micro_all <- bind_rows(
  train_df %>% mutate(split = "train"),
  test_df  %>% mutate(split = "test")
)

###############################################################
# 2) Basic BMI summaries (from df_micro_all)
###############################################################

bmi_summary <- df_micro_all %>%
  summarise(
    n = n(),
    bmi_mean = mean(bmi, na.rm = TRUE),
    bmi_sd = sd(bmi, na.rm = TRUE),
    bmi_median = median(bmi, na.rm = TRUE),
    bmi_iqr = IQR(bmi, na.rm = TRUE),
    bmi_min = min(bmi, na.rm = TRUE),
    bmi_max = max(bmi, na.rm = TRUE)
  )

write_csv(bmi_summary, file.path(RESULTS_DIR, "bmi_summary.csv"))
print(bmi_summary)

# BMI histogram
p_bmi_hist <- ggplot(df_micro_all, aes(x = bmi)) +
  geom_histogram(bins = 40) +
  labs(
    title = "BMI distribution (microbiome dataset)",
    x = "BMI",
    y = "Count"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "bmi_histogram.png"), p_bmi_hist, width = 7, height = 5, dpi = 300)
print(p_bmi_hist)

# BMI density by split
p_bmi_density <- ggplot(df_micro_all, aes(x = bmi, color = split)) +
  geom_density(linewidth = 1.1) +
  labs(
    title = "BMI density by split",
    x = "BMI",
    y = "Density",
    color = "Split"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "bmi_density_by_split.png"), p_bmi_density, width = 7, height = 5, dpi = 300)
print(p_bmi_density)

###############################################################
# 3) BMI classes (WHO cutoffs) — optional but useful for figures
###############################################################

df_micro_all <- df_micro_all %>%
  mutate(
    bmi_class = case_when(
      bmi < 18.5 ~ "Underweight",
      bmi < 25   ~ "Normal",
      bmi < 30   ~ "Overweight",
      TRUE       ~ "Obese"
    ),
    bmi_class = factor(bmi_class, levels = c("Underweight", "Normal", "Overweight", "Obese"))
  )

bmi_class_table <- df_micro_all %>%
  count(bmi_class) %>%
  mutate(prop = n / sum(n))

write_csv(bmi_class_table, file.path(RESULTS_DIR, "bmi_class_counts.csv"))
print(bmi_class_table)

p_bmi_class <- ggplot(df_micro_all, aes(x = bmi_class)) +
  geom_bar() +
  labs(
    title = "BMI class distribution",
    x = NULL,
    y = "Count"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "bmi_class_distribution.png"), p_bmi_class, width = 7, height = 5, dpi = 300)
print(p_bmi_class)

###############################################################
# 4) Metadata plots (age/sex) if md available
###############################################################


# Try common column names
# You said: age_years and sex are in the ML splits, but md might have different naming.
# We'll prefer age_years/sex if they exist in md.
age_col <- intersect(c("age_years", "age", "host_age", "subject_age"), colnames(md))[1]
sex_col <- intersect(c("sex", "gender"), colnames(md))[1]
  

p_age <- ggplot(md, aes(x = .data[[age_col]])) +
  geom_histogram(bins = 40) +
  labs(
    title = "Age distribution (metadata)",
    x = "Age",
    y = "Count"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "age_histogram.png"), p_age, width = 7, height = 5, dpi = 300)
print(p_age)
  

p_sex <- ggplot(md, aes(x = .data[[sex_col]])) +
  geom_bar() +
  labs(
    title = "Sex distribution (metadata)",
    x = NULL,
    y = "Count"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "sex_distribution.png"), p_sex, width = 7, height = 5, dpi = 300)
print(p_sex)

  
# BMI vs Age
p_bmi_age <- ggplot(md, aes(x = .data[[age_col]], y = bmi)) +
  geom_point(alpha = 0.3, size = 0.8) +
  geom_smooth(method = "loess", se = FALSE) +
  labs(
    title = "BMI vs age (metadata)",
    x = "Age",
    y = "BMI"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "bmi_vs_age.png"), p_bmi_age, width = 7, height = 5, dpi = 300)
print(p_bmi_age)
  
}

###############################################################
# 5) Top-20 Spearman correlations (microbiome features vs BMI)
###############################################################

# Exclude non-feature columns:
feature_cols <- setdiff(colnames(df_micro_all), c("bmi", "split", "bmi_class"))

# Safety: if too many features, you can sample features or skip

cors <- map_dbl(
  feature_cols,
  ~ suppressWarnings(cor(df_micro_all[[.x]], df_micro_all$bmi, method = "spearman"))
)

names(cors) <- feature_cols

cors_df <- tibble(
  feature = names(cors),
  spearman = as.numeric(cors)
) %>%
  arrange(desc(abs(spearman))) %>%
  slice(1:20)

write_csv(cors_df, file.path(RESULTS_DIR, "top20_spearman_microbiome_vs_bmi.csv"))

p_spear <- ggplot(cors_df, aes(x = reorder(feature, spearman), y = spearman)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 20 Spearman correlations: microbiome features vs BMI",
    x = NULL,
    y = "Spearman correlation"
  ) +
  theme_minimal(base_size = 13)

ggsave(file.path(PLOTS_DIR, "top20_spearman_microbiome_vs_bmi.png"), p_spear, width = 8, height = 6, dpi = 300)
print(p_spear)

