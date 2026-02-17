###############################################################
# plots/shap_comparison.R
# Compare SHAP importances across models (MICROBIOME):
# - Top-20 per model barplot (faceted)
# - Spearman correlation of importance rankings across models
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)

SHAP_DIR <- file.path(RESULTS_DIR, "shap")
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

paths <- c(
  file.path(SHAP_DIR, "shap_importance_xgboost_microbiome.csv"),
  file.path(SHAP_DIR, "shap_importance_catboost_microbiome.csv"),
  file.path(SHAP_DIR, "shap_importance_tabpfn_microbiome.csv")
)

available <- paths[file.exists(paths)]
if (length(available) == 0) stop("No SHAP importance files found in: ", SHAP_DIR)

imp_df <- bind_rows(lapply(available, read_csv, show_col_types = FALSE)) %>%
  mutate(model = factor(model, levels = c("XGBoost", "CatBoost", "TabPFN")))

###############################################################
# Helpers: reorder within facets (no extra packages)
###############################################################
reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun)
}
scale_x_reordered <- function(..., sep = "___") {
  ggplot2::scale_x_discrete(labels = function(x) gsub(paste0(sep, ".*$"), "", x), ...)
}

###############################################################
# 1) Top-20 barplot per model
###############################################################

topN <- 20

plot_df <- imp_df %>%
  group_by(model) %>%
  slice_max(mean_abs_shap, n = topN, with_ties = FALSE) %>%
  ungroup() %>%
  group_by(model) %>%
  mutate(feature = reorder_within(feature, mean_abs_shap, model)) %>%
  ungroup()

p <- ggplot(plot_df, aes(x = feature, y = mean_abs_shap)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ model, scales = "free_y") +
  scale_x_reordered() +
  labs(
    title = "SHAP feature importance comparison (microbiome-only)",
    x = NULL,
    y = "Mean |SHAP|"
  ) +
  theme_minimal(base_size = 12)

ggsave(
  file.path(PLOTS_DIR, "shap_comparison_top20_microbiome.png"),
  p,
  width = 10,
  height = 7,
  dpi = 300
)

print(p)

###############################################################
# 2) Spearman correlation of importance across models
###############################################################

wide <- imp_df %>%
  select(model, feature, mean_abs_shap) %>%
  pivot_wider(names_from = model, values_from = mean_abs_shap)

models <- intersect(c("XGBoost", "CatBoost", "TabPFN"), colnames(wide))

cor_mat <- matrix(NA_real_, nrow = length(models), ncol = length(models),
                  dimnames = list(models, models))

for (i in seq_along(models)) {
  for (j in seq_along(models)) {
    a <- wide[[models[i]]]
    b <- wide[[models[j]]]
    cor_mat[i, j] <- suppressWarnings(cor(a, b, method = "spearman", use = "complete.obs"))
  }
}

cor_df <- as.data.frame(as.table(cor_mat)) %>%
  transmute(model_a = Var1, model_b = Var2, spearman = Freq)

write_csv(cor_df, file.path(RESULTS_DIR, "shap_importance_spearman_microbiome.csv"))

message("Saved:\n - shap_comparison_top20_microbiome.png\n - shap_importance_spearman_microbiome.csv")