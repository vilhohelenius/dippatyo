###############################################################
# analysis/pcoa_beta_diversity.R
# PCoA (beta diversity) + PERMANOVA using microbiome-only dataset
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(ggplot2)
library(vegan)
library(ape)
library(tibble)

dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################
# Load microbiome-only data (train + test)
###############################################################

train_path <- file.path(EXPORT_DIR, "train_microbiome.csv")
test_path  <- file.path(EXPORT_DIR, "test_microbiome.csv")

if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("Missing export files:\n- ", train_path, "\n- ", test_path)
}

train_df <- read_csv(train_path, show_col_types = FALSE) %>% mutate(split = "train")
test_df  <- read_csv(test_path,  show_col_types = FALSE) %>% mutate(split = "test")

df <- bind_rows(train_df, test_df)

###############################################################
# BMI classes (WHO cutoffs)
###############################################################

df <- df %>%
  mutate(
    bmi_class = case_when(
      bmi < 18.5 ~ "Underweight",
      bmi < 25   ~ "Normal",
      bmi < 30   ~ "Overweight",
      TRUE       ~ "Obese"
    ),
    bmi_class = factor(bmi_class, levels = c("Underweight", "Normal", "Overweight", "Obese"))
  )

###############################################################
# Feature matrix
###############################################################

X <- df %>% select(-bmi, -split, -bmi_class)
X_mat <- as.matrix(X)

###############################################################
# Bray–Curtis distance + PCoA
###############################################################

# Bray-Curtis expects non-negative inputs; CLR can be negative.
# Still usable as a distance calculation in practice, but if you prefer:
# - Use Euclidean on CLR: dist(X_mat)
# - Or compute Bray on raw relative abundances before CLR
# Here we follow your earlier approach (Bray), but warn if negatives exist.
if (any(X_mat < 0, na.rm = TRUE)) {
  warning("CLR features contain negative values. Bray–Curtis is traditionally used on non-negative abundances.\n",
          "If you want a fully standard approach: use Euclidean distance on CLR (dist(X_mat)).")
}

bray_dist <- vegan::vegdist(dist(X_mat), method = "bray")

pcoa_res <- ape::pcoa(bray_dist)

pcoa_df <- as.data.frame(pcoa_res$vectors[, 1:2])
colnames(pcoa_df) <- c("PCoA1", "PCoA2")

# % explained
eig <- pcoa_res$values$Relative_eig
var1 <- round(100 * eig[1], 2)
var2 <- round(100 * eig[2], 2)

pcoa_df <- pcoa_df %>%
  mutate(
    bmi = df$bmi,
    bmi_class = df$bmi_class,
    split = df$split
  )

###############################################################
# Plot
###############################################################

p <- ggplot(pcoa_df, aes(x = PCoA1, y = PCoA2, color = bmi_class)) +
  geom_point(alpha = 0.6, size = 1.2) +
  labs(
    title = "PCoA of microbiome profiles (Bray–Curtis)",
    subtitle = "Colored by BMI class",
    x = paste0("PCoA1 (", var1, "%)"),
    y = paste0("PCoA2 (", var2, "%)"),
    color = "BMI class"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  file.path(PLOTS_DIR, "pcoa_bray_bmi_class_microbiome.png"),
  p,
  width = 7,
  height = 5,
  dpi = 300
)

print(p)

###############################################################
# PERMANOVA (adonis2)
###############################################################

# adonis2 expects a data frame with the predictor variable
adonis_res <- vegan::adonis2(
  bray_dist ~ bmi_class,
  data = df,
  permutations = 999
)

# Save PERMANOVA output
sink(file.path(RESULTS_DIR, "permanova_bmi_class_microbiome.txt"))
print(adonis_res)
sink()

message("Saved:\n - ", file.path(PLOTS_DIR, "pcoa_bray_bmi_class_microbiome.png"),
        "\n - ", file.path(RESULTS_DIR, "permanova_bmi_class_microbiome.txt"))