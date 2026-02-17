###############################################################
# load_preprocess.R
# Load Metalog microbiome data and preprocess for ML
###############################################################

source("/Users/vilhohelenius/Documents/Kouluhommat/Dippatyö/r_koodi/config.R")

library(dplyr)
library(readr)
library(tidyr)
library(httr)
library(compositions)

###############################################################
# Helper — download Metalog file
###############################################################

download_if_missing <- function(base_url, download_dir = DATA_DIR) {
  
  dir.create(download_dir, showWarnings = FALSE)
  
  base_filename <- basename(base_url)
  pattern <- sub("latest", "[0-9]{4}-[0-9]{2}-[0-9]{2}", base_filename)
  
  existing <- list.files(download_dir, pattern = pattern, full.names = TRUE)
  if (length(existing) > 0) return(max(existing))
  
  message("Downloading: ", base_url)
  
  response <- httr::GET(base_url, followlocation = TRUE)
  if (httr::status_code(response) != 200) {
    stop("Download failed: ", httr::status_code(response))
  }
  
  dated_url <- response$url
  filename <- basename(dated_url)
  destfile <- file.path(download_dir, filename)
  
  httr::GET(dated_url, httr::write_disk(destfile, overwrite = TRUE))
  
  return(destfile)
}

###############################################################
# 1. Load metadata
###############################################################

metadata_url <- "https://metalog.embl.de/static/download/metadata/human_extended_wide_latest.tsv.gz"
md_file <- download_if_missing(metadata_url)

md <- read_tsv(md_file, guess_max = Inf, show_col_types = FALSE)

###############################################################
# Metadata filtering
###############################################################

md <- md %>%
  filter(
    environment_material == "fecal material [ENVO:00002003]",
    age_category == "adult",
    timeseries_available == "no",
    is.na(artificial),
    !is.na(bmi),
    !is.na(age_years),
    !is.na(sex)
  )

# Ensure sex is numeric (0/1) for all models
md <- md %>%
  mutate(sex = case_when(
    tolower(sex) %in% c("male", "m")   ~ 1,
    tolower(sex) %in% c("female", "f") ~ 0,
    TRUE ~ NA_real_
  )) %>%
  filter(!is.na(sex))   # drop samples with unknown sex

message("Metadata samples after filtering: ", nrow(md))

###############################################################
# Save filtered metadata (for descriptive analysis etc.)
###############################################################

saveRDS(md, file.path(DATA_DIR, "md.rds"))
write_csv(md, file.path(DATA_DIR, "md.csv"))

message("Saved filtered metadata:")
message(" - data/md.rds")
message(" - data/md.csv")

###############################################################
# 2. Load MetaPhlAn4 species profiles
###############################################################

taxa_url <- "https://metalog.embl.de/static/download/profiles/human_metaphlan4_species_latest.tsv.gz"
taxa_file <- download_if_missing(taxa_url)

taxa <- read_tsv(taxa_file, show_col_types = FALSE) %>%
  mutate(rel_abund = as.numeric(rel_abund))

###############################################################
# Keep only matching samples
###############################################################

taxa <- taxa %>% semi_join(md, by = "sample_alias")
md   <- md   %>% semi_join(taxa, by = "sample_alias")

message("Samples after matching: ", n_distinct(taxa$sample_alias))

###############################################################
# 3. Prevalence filtering (≥25 % samples)
###############################################################

N_samples <- n_distinct(taxa$sample_alias)
cutoff <- 0.25 * N_samples

taxa <- taxa %>%
  group_by(species) %>%
  filter(n() >= cutoff) %>%
  ungroup()

message("Species after prevalence filtering: ", n_distinct(taxa$species))

###############################################################
# 4. Convert to wide format
###############################################################

taxa_wide <- taxa %>%
  pivot_wider(names_from = species, values_from = rel_abund, values_fill = 0)

taxa_df <- as.data.frame(taxa_wide)
rownames(taxa_df) <- taxa_df$sample_alias
taxa_df$sample_alias <- NULL

###############################################################
# 5. CLR transformation
###############################################################

taxa_clr <- compositions::clr(as.matrix(taxa_df) + 1e-6)

df_ml <- as.data.frame(taxa_clr)
df_ml$sample_alias <- rownames(df_ml)

###############################################################
# 6. Add metadata (BMI, age, sex)
###############################################################

df_ml <- df_ml %>%
  left_join(
    md %>% select(sample_alias, bmi, age_years, sex),
    by = "sample_alias"
  )

rownames(df_ml) <- df_ml$sample_alias
df_ml$sample_alias <- NULL

###############################################################
# Final dataset summary
###############################################################

message("Final ML dataset:")
message("Samples:  ", nrow(df_ml))
message("Features: ", ncol(df_ml) - 3, " microbiome species")
message("Metadata: BMI + age + sex")

###############################################################
# Save dataset
###############################################################

dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

saveRDS(df_ml, file.path(DATA_DIR, "df_ml_full.rds"))
write_csv(df_ml, file.path(DATA_DIR, "df_ml_full.csv"))

message("Saved:")
message(" - data/df_ml_full.rds (R cache)")
message(" - data/df_ml_full.csv (CSV snapshot)")