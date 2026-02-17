###############################################################
# plots/scaling_plot.R
# Plot learning curves (MICROBIOME, fixed test set)
# Uses combined results if available (R models + TabPFN).
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(ggplot2)
library(scales)

dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# Load scaling results
###############################################################

combined_path <- file.path(RESULTS_DIR, "scaling_results_combined_test_microbiome.csv")
r_only_path   <- file.path(RESULTS_DIR, "scaling_results_r_test_microbiome.csv")
tabpfn_path_1 <- file.path(RESULTS_DIR, "tabpfn", "scaling_results_microbiome.csv")
tabpfn_path_2 <- file.path(RESULTS_DIR, "tabpfn", "scaling_results.csv")

if (file.exists(combined_path)) {
  df <- read_csv(combined_path, show_col_types = FALSE)
  message("Loaded combined scaling results: ", combined_path)
  
} else if (file.exists(r_only_path)) {
  df <- read_csv(r_only_path, show_col_types = FALSE)
  message("Loaded R-only scaling results: ", r_only_path)
  
  # Try append TabPFN if available
  tabpfn_path <- c(tabpfn_path_1, tabpfn_path_2)[file.exists(c(tabpfn_path_1, tabpfn_path_2))][1]
  if (!is.na(tabpfn_path) && file.exists(tabpfn_path)) {
    tabpfn_df <- read_csv(tabpfn_path, show_col_types = FALSE) %>%
      transmute(
        n = as.numeric(n),
        model = "TabPFN",
        R2 = as.numeric(R2),
        RMSE = as.numeric(RMSE),
        MAE = as.numeric(MAE),
        dataset = "microbiome"
      )
    df <- bind_rows(df, tabpfn_df)
    message("Appended TabPFN scaling results: ", tabpfn_path)
  } else {
    warning("No TabPFN scaling file found. Plot will include only R models.")
  }
  
} else {
  stop("No scaling results found. Expected one of:\n- ", combined_path, "\n- ", r_only_path)
}

###############################################################
# Clean / harmonize
###############################################################

df <- df %>%
  mutate(
    n = as.numeric(n),
    model = factor(model, levels = c("Lasso", "XGBoost", "CatBoost", "TabPFN"))
  ) %>%
  arrange(model, n)

###############################################################
# Plot: R2
###############################################################

p_r2 <- ggplot(df, aes(x = n, y = R2, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  scale_x_log10(breaks = unique(df$n)) +
  annotation_logticks(sides = "b") +
  labs(
    title = "Learning curves (microbiome): performance vs training set size",
    x = "Training set size (log scale)",
    y = expression(R^2),
    color = "Model"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  file.path(PLOTS_DIR, "scaling_microbiome_R2.png"),
  p_r2,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# Plot: RMSE
###############################################################

p_rmse <- ggplot(df, aes(x = n, y = RMSE, color = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  scale_x_log10(breaks = unique(df$n)) +
  annotation_logticks(sides = "b") +
  labs(
    title = "Learning curves (microbiome): RMSE vs training set size",
    x = "Training set size (log scale)",
    y = "RMSE (lower is better)",
    color = "Model"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  file.path(PLOTS_DIR, "scaling_microbiome_RMSE.png"),
  p_rmse,
  width = 7,
  height = 5,
  dpi = 300
)

print(p_r2)
print(p_rmse)


cat("\nSaved plots:\n")
cat(" -", file.path(PLOTS_DIR, "scaling_microbiome_R2.png"), "\n")
cat(" -", file.path(PLOTS_DIR, "scaling_microbiome_RMSE.png"), "\n")
