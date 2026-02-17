source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(tibble)
library(purrr)
library(ggplot2)
library(yardstick)

dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Helper: load predictions
# -------------------------
load_preds <- function(path, model_name) {
  if (!file.exists(path)) {
    message("Missing predictions file for ", model_name, ": ", path)
    return(NULL)
  }
  df <- read_csv(path, show_col_types = FALSE)
  
  # Accept either (y_true, y_pred) or (bmi_true, bmi_pred) etc.
  # Try to standardize column names
  cols <- colnames(df)
  
  if (!("y_true" %in% cols)) {
    true_candidates <- c("bmi_true", "true", "actual", "y")
    hit <- intersect(true_candidates, cols)
    if (length(hit) > 0) df <- df %>% rename(y_true = all_of(hit[1]))
  }
  if (!("y_pred" %in% cols)) {
    pred_candidates <- c("bmi_pred", "pred", "prediction", "yhat")
    hit <- intersect(pred_candidates, cols)
    if (length(hit) > 0) df <- df %>% rename(y_pred = all_of(hit[1]))
  }
  
  stopifnot("y_true" %in% colnames(df), "y_pred" %in% colnames(df))
  
  df %>%
    transmute(
      model = model_name,
      y_true = as.numeric(y_true),
      y_pred = as.numeric(y_pred)
    )
}

# -------------------------
# Collect predictions
# -------------------------
pred_paths <- list(
  Lasso   = "results/lasso/predictions_test.csv",
  XGBoost = "results/xgboost/predictions_test.csv",
  CatBoost= "results/catboost/predictions_test.csv",
  TabPFN  = "results/tabpfn/predictions_test.csv"   # <-- put your TabPFN file here
)

pred_list <- imap(pred_paths, load_preds) %>% discard(is.null)

pred_all <- bind_rows(pred_list)

if (nrow(pred_all) == 0) stop("No prediction files found. Cannot create model comparison.")

# -------------------------
# Compute metrics per model
# -------------------------
metrics_df <- pred_all %>%
  group_by(model) %>%
  summarise(
    R2   = rsq_vec(y_true, y_pred),
    RMSE = rmse_vec(y_true, y_pred),
    MAE  = mae_vec(y_true, y_pred),
    .groups = "drop"
  ) %>%
  arrange(RMSE)

write_csv(metrics_df, file.path(RESULTS_DIR, "model_metrics_test.csv"))

# Create label for facet titles
metrics_lab <- metrics_df %>%
  mutate(label = sprintf("R²=%.3f | RMSE=%.3f | MAE=%.3f", R2, RMSE, MAE)) %>%
  select(model, label)

pred_plot_df <- pred_all %>%
  left_join(metrics_lab, by = "model") %>%
  mutate(model_facet = paste0(model, "\n", label))

# -------------------------
# Plot: Predicted vs Actual
# -------------------------
p <- ggplot(pred_plot_df, aes(x = y_true, y = y_pred)) +
  geom_point(alpha = 0.25, size = 0.8) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~ model_facet, scales = "free", ncol = 2) +
  labs(
    title = "Predicted vs Actual BMI (test set)",
    x = "Actual BMI",
    y = "Predicted BMI"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(PLOTS_DIR, "model_comparison_pred_vs_actual.png"),
  plot = p,
  width = 9,
  height = 7,
  dpi = 300
)

print(p)

cat("\nSaved:\n")
cat(" -", file.path(PLOTS_DIR, "model_comparison_pred_vs_actual.png"), "\n")
cat(" -", file.path(RESULTS_DIR, "model_metrics_test.csv"), "\n")