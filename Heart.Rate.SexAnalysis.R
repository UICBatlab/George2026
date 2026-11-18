# Load necessary libraries
library(ggplot2)
library(dplyr)
library(viridis)
library(ggpubr)
library(FSA)

# Define folder paths for females and males
female_folder_path <- "/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Females"
male_folder_path <- "/Users/zariageorge/Desktop/WNPLAY/6.05.playlist/Males"

# Get all CSV files from both folders
female_csv_files <- list.files(female_folder_path, pattern = "\\.csv$", full.names = TRUE)
male_csv_files <- list.files(male_folder_path, pattern = "\\.csv$", full.names = TRUE)

# Define original time windows (stimulus onset times)
time_windows <- list(
  Aggression = list(c(61.302,90.813), c(691.400,720.914), c(871.504,901.015),
                    c(961.503,991.014), c(1231.604,1261.114), c(1411.604,1441.113)),
  Distress   = list(c(151.302,180.851), c(241.303,270.072), c(601.403,630.953),
                    c(1051.501,1081.052), c(1951.803,1981.352), c(2041.804,2071.353)),
  Echolocation = list(c(421.402,450.44), c(511.746,540.944), c(1321.602,1351.145),
                      c(1501.602,1531.146), c(1771.803,1801.345), c(1861.803,1891.345)),
  Distortion = list(c(2131.428,2161.605), c(331.302,361.104), c(781.401,811.205),
                    c(1141.503,1171.305), c(1591.603,1621.406), c(1681.703,1711.504))
)

alternative_time_windows <- list(
  Aggression   = list(c(60.189,90.189), c(150.189,180.189), c(510.389,540.389),
                      c(1590.790,1620.790), c(1770.790,1800.790), c(2130.791,2160.791)),
  Distress     = list(c(420.389,450.389), c(600.389,630.389), c(780.489,810.489),
                      c(870.489,900.489), c(1050.490,1080.490), c(1950.790,1980.790)),
  Echolocation = list(c(960.490,990.490), c(1320.690,1350.690), c(1500.790,1530.790),
                      c(1680.790,1710.790), c(1860.790,1890.790), c(2040.790,2070.790)),
  Distortion   = list(c(691.400,720.914), c(1141.503,1171.305), c(1231.604,1261.114),
                      c(1411.604,1441.113), c(241.303,270.072), c(331.302,361.104))
)

# Function to get the appropriate time windows for a bat
get_time_windows <- function(bat_id) {
  if (bat_id %in% c("AF4", "AF6", "B32", "AD6")) {
    return(alternative_time_windows)
  } else {
    return(time_windows)
  }
}

# Function to calculate robust baseline heart rate
calculate_robust_baseline <- function(df) {
  valid_rows <- df$Heart_Rate > 0 & df$Heart_Rate <= 350

  valid_hr <- df$Heart_Rate[valid_rows]
  
  if (length(valid_hr) < 5) {
    return(NA)
  }
  
  window_size <- 30
  min_avg_hr <- Inf
  
  time_range <- seq(min(df$Time), max(df$Time) - window_size, by = 5)
  
  for (start_time in time_range) {
    window_rows <- df$Time >= start_time & 
      df$Time <= start_time + window_size &
      df$Heart_Rate > 0 &
      df$Heart_Rate <= 310
    window_hr <- df$Heart_Rate[window_rows]
    
    if (length(window_hr) >= 5) {
      avg_hr <- mean(window_hr)
      if (avg_hr < min_avg_hr) {
        min_avg_hr <- avg_hr
      }
    }
  }
  
  percentile_baseline <- quantile(valid_hr, 0.05)
  
  if (min_avg_hr < 50 || is.infinite(min_avg_hr)) {
    return(percentile_baseline)
  }
  
  return(min(min_avg_hr, percentile_baseline))
}

# Initialize lists to store data
plot_data <- list()
baseline_comparison <- list()

# Process all files
for (file_info in list(list(files = female_csv_files, gender = "Female"),
                       list(files = male_csv_files, gender = "Male"))) {
  
  csv_files <- file_info$files
  gender_label <- file_info$gender
  
  for (file in csv_files) {
    tryCatch({
      df <- read.csv(file)
      file_name <- basename(file)
      bat_id <- substr(file_name, 1, 3)
      
      # Calculate original baseline
      baseline_rows <- df$File_Marker == 0 & df$Heart_Rate > 0 & df$Heart_Rate <= 310
      original_baseline <- mean(df$Heart_Rate[baseline_rows], na.rm = TRUE)
      
      # Calculate robust baseline
      robust_baseline <- calculate_robust_baseline(df)
      
      if (!is.na(robust_baseline)) {
        baseline_hr <- robust_baseline
        baseline_method <- "robust"
      } else {
        baseline_hr <- original_baseline
        baseline_method <- "original"
      }
      
      # Record baseline comparison
      baseline_comparison[[length(baseline_comparison) + 1]] <- data.frame(
        Bat = bat_id,
        File = file_name,
        Original_Baseline = original_baseline,
        Robust_Baseline = baseline_hr,
        Method_Used = baseline_method,
        Difference = ifelse(baseline_method != "original", original_baseline - baseline_hr, 0)
      )
      
      # Normalize time
      marker_rows <- df$File_Marker == 1
      if (any(marker_rows)) {
        first_marker_time <- min(df$Time[marker_rows], na.rm = TRUE)
        df$Time <- df$Time - first_marker_time
      }
      
      # Get appropriate time windows for this bat
      bat_time_windows <- get_time_windows(bat_id)
      
      # Process each category
      for (category in names(bat_time_windows)) {
        intervals <- bat_time_windows[[category]]
        hr_values <- c()
        
        for (interval in intervals) {
          start <- interval[1]
          end <- interval[2]
          
          segment_rows <- df$Time >= start & df$Time <= end & 
            df$Heart_Rate > 0 & df$Heart_Rate <= 500
          hr_segment <- df$Heart_Rate[segment_rows]
          
          hr_values <- c(hr_values, hr_segment)
        }
        
        if (length(hr_values) > 0) {
          avg_hr <- mean(hr_values)
          change_from_baseline <- avg_hr - baseline_hr
          
          plot_data[[length(plot_data) + 1]] <- data.frame(
            Bat_ID = bat_id,
            Stimulus = category,
            Change_from_Baseline = change_from_baseline,
            Sex = gender_label,
            Avg_Heart_Rate = avg_hr,
            Baseline_HR = baseline_hr
          )
        }
      }
      
    }, error = function(e) {
      message(paste("Error processing file", file, ":", e$message))
    })
  }
}

# Convert lists to data frames
HRData_Clean <- do.call(rbind, plot_data)
baseline_df <- do.call(rbind, baseline_comparison)

# Print summary of baseline differences
cat("\nBaseline Comparison Summary:\n")
cat(sprintf("Average difference between original and robust baseline: %.2f BPM\n", 
            mean(baseline_df$Difference, na.rm = TRUE)))
cat(sprintf("Max difference: %.2f BPM\n", 
            max(baseline_df$Difference, na.rm = TRUE)))

# Filter data
filter_rows <- HRData_Clean$Change_from_Baseline > 0 & 
  HRData_Clean$Change_from_Baseline <= 300
HRData_Clean <- HRData_Clean[filter_rows, ]

# Summary statistics using aggregate
HR_summary <- aggregate(
  Change_from_Baseline ~ Stimulus,
  data = HRData_Clean,
  FUN = function(x) {
    c(n = length(x),
      Mean_Change = mean(x, na.rm = TRUE),
      SD_Change = sd(x, na.rm = TRUE),
      Median_Change = median(x, na.rm = TRUE),
      Min_Change = min(x, na.rm = TRUE),
      Max_Change = max(x, na.rm = TRUE))
  }
)
print(HR_summary)

# Check distribution visually
qqnorm(HRData_Clean$Change_from_Baseline, main = "QQ Plot")
qqline(HRData_Clean$Change_from_Baseline, col = "red", lwd = 2)

hist(HRData_Clean$Change_from_Baseline,
     main = "Histogram of HR Change from Baseline",
     xlab = "Change in Heart Rate (BPM)",
     col = "blue", border = "white")

# Plotting - Boxplot of change from baseline by stimulus
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.4, size = 2) +
  labs(
    title = "Change in Heart Rate from Baseline by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_viridis_d(option = "rocket")

# Calculate mean change for each bat within each stimulus
Bat_Means_Overall <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = HRData_Clean,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Bat_Means_Overall)[3] <- "Mean_Change_per_Bat"

# Box plot with mean as points
ggplot(HRData_Clean, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_point(
    data = Bat_Means_Overall,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  labs(
    title = "Change in Heart Rate from Baseline by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_viridis_d(option = "rocket")

# Non-parametric test: Kruskal-Wallis
kruskal_test <- kruskal.test(Change_from_Baseline ~ Stimulus, data = HRData_Clean)
print(kruskal_test)

# Add Sex variable
HRData_Clean$Sex <- NA
HRData_Clean$Sex[HRData_Clean$Bat_ID %in% c("A88", "AEO", "B05", "B1D","AD6","AF4")] <- "Male"
HRData_Clean$Sex[HRData_Clean$Bat_ID %in% c("693", "B04", "CD3", "B30","B32",'AF6')] <- "Female"

# Remove rows with NA Sex
HRData_Clean <- HRData_Clean[!is.na(HRData_Clean$Sex), ]

# Mann-Whitney U test (Wilcoxon test)

wilcox_test_result <- wilcox.test(Change_from_Baseline ~ Sex, data = HRData_Clean)
print(wilcox_test_result)

ggplot(HRData_Clean, aes(x = Sex, y = Change_from_Baseline, fill = Sex)) +
  geom_boxplot(alpha = 0.4, outlier.shape = NA) +
  geom_jitter(
    aes(color = Sex),
    position = position_jitterdodge(jitter.width = 0.6, dodge.width = 0.8),
    size = 2,
    alpha = 0.7
  ) +
  stat_compare_means(
    method = "t.test",
    comparisons = list(c("Male", "Female")),
    label = "p.signif",
    bracket.size = 0.8,
    label.y = max(HRData_Clean$Change_from_Baseline, na.rm = TRUE) + 5
  ) +
  labs(
    title = "Overall Change in Heart Rate by Sex",
    x = "Sex",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  scale_color_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))


# ── Sex differences in heart rate using ONE value per bat ────────────────────
# Run after HRData_Clean is built and Sex is assigned.
library(dplyr)
library(ggplot2)

sex_colors <- c("Male" = "#FFA500", "Female" = "#800080")

fmt_p <- function(p) ifelse(p < 0.001, "p < 0.001", sprintf("p = %.3f", p))
p_star <- function(p) ifelse(p < 0.001, "***", ifelse(p < 0.01, "**",
                                                      ifelse(p < 0.05, "*", "ns")))

# ── 1. Overall: one median per bat across all stimuli ─────────────────────────
per_bat <- HRData_Clean %>%
  group_by(Bat_ID, Sex) %>%
  summarise(Median_Change = median(Change_from_Baseline, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(Sex = factor(Sex, levels = c("Female", "Male")))

cat("\n=== Bats per sex ===\n")
print(table(per_bat$Sex))          # should be 6 and 6

overall_test <- wilcox.test(Median_Change ~ Sex, data = per_bat, exact = TRUE)
overall_meds <- per_bat %>% group_by(Sex) %>%
  summarise(Median = median(Median_Change), n = n(), .groups = "drop")

cat("\n=== Overall sex difference (per-bat medians) ===\n")
print(overall_test)
print(overall_meds)

# ── 2. Per stimulus: one median per bat per stimulus (Figure 2d) ──────────────
per_bat_stim <- HRData_Clean %>%
  group_by(Bat_ID, Sex, Stimulus) %>%
  summarise(Median_Change = median(Change_from_Baseline, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(Sex = factor(Sex, levels = c("Female", "Male")))

stim_results <- per_bat_stim %>%
  group_by(Stimulus) %>%
  summarise(
    n_Female      = sum(Sex == "Female"),
    n_Male        = sum(Sex == "Male"),
    Median_Female = median(Median_Change[Sex == "Female"]),
    Median_Male   = median(Median_Change[Sex == "Male"]),
    W             = unname(wilcox.test(Median_Change ~ Sex, exact = TRUE)$statistic),
    p             = wilcox.test(Median_Change ~ Sex, exact = TRUE)$p.value,
    .groups = "drop"
  ) %>%
  mutate(p_BH = p.adjust(p, method = "BH"),   # optional: corrected for 4 tests
         sig  = p_star(p))

cat("\n=== Sex difference within each stimulus (per-bat medians) ===\n")
print(as.data.frame(stim_results), digits = 3)

# ── 3. Ready-to-paste sentences ───────────────────────────────────────────────
fm <- overall_meds$Median[overall_meds$Sex == "Female"]
mm <- overall_meds$Median[overall_meds$Sex == "Male"]
cat(sprintf(paste0(
  "\nOverall heart rate change %s between sexes (Wilcoxon rank-sum test on per-bat ",
  "medians, W = %.0f, %s; females: median = %.1f BPM, males: median = %.1f BPM; ",
  "n = 6 per sex).\n"),
  ifelse(overall_test$p.value < 0.05, "differed significantly", "did not differ significantly"),
  overall_test$statistic, fmt_p(overall_test$p.value), fm, mm))

for (i in seq_len(nrow(stim_results))) {
  r <- stim_results[i, ]
  cat(sprintf("%s: W = %.0f, %s (females %.1f vs males %.1f BPM)\n",
              r$Stimulus, r$W, fmt_p(r$p), r$Median_Female, r$Median_Male))
}

# ── 4. Figure 2c: overall sex difference, one dot per bat ────────────────────
ymax <- max(per_bat$Median_Change, na.rm = TRUE)

p_overall <- ggplot(per_bat, aes(x = Sex, y = Median_Change, fill = Sex)) +
  geom_boxplot(width = 0.5, alpha = 0.6, outlier.shape = NA) +
  geom_point(aes(color = Sex), size = 3,
             position = position_jitter(width = 0.08, seed = 1)) +
  annotate("segment", x = 1, xend = 2, y = ymax * 1.08, yend = ymax * 1.08,
           linewidth = 0.8) +
  annotate("text", x = 1.5, y = ymax * 1.13,
           label = p_star(overall_test$p.value), size = 4.5, fontface = "bold") +
  scale_fill_manual(values = sex_colors) +
  scale_color_manual(values = sex_colors) +
  labs(x = "Sex", y = "Δ Heart Rate (BPM)",
       subtitle = "Wilcoxon rank-sum on per-bat medians, n = 6 per sex") +
  coord_cartesian(ylim = c(NA, ymax * 1.18), clip = "off") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none", aspect.ratio = 1.6,
        plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10))
print(p_overall)

# ── 5. Figure 2d: sex difference per stimulus, one dot per bat ───────────────
ymax_s <- max(per_bat_stim$Median_Change, na.rm = TRUE)
lab_df <- stim_results %>% mutate(y = ymax_s * 1.08)

p_stim <- ggplot(per_bat_stim, aes(x = Stimulus, y = Median_Change, fill = Sex)) +
  geom_boxplot(width = 0.5, position = position_dodge(width = 0.7),
               outlier.shape = NA, alpha = 0.6) +
  geom_point(aes(color = Sex), size = 2.5,
             position = position_jitterdodge(jitter.width = 0.1,
                                             dodge.width = 0.7, seed = 1)) +
  geom_text(data = lab_df, aes(x = Stimulus, y = y, label = sig),
            inherit.aes = FALSE, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = sex_colors) +
  scale_color_manual(values = sex_colors) +
  labs(x = "Stimulus", y = "Δ Heart Rate (BPM)", fill = "Sex", color = "Sex",
       subtitle = "Wilcoxon rank-sum on per-bat medians, n = 6 per sex") +
  coord_cartesian(ylim = c(NA, ymax_s * 1.15), clip = "off") +
  theme_minimal(base_size = 14) +
  theme(plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10))
print(p_stim)
#test end

#individual sex 
# Separate data by sex
Male_Data <- HRData_Clean[HRData_Clean$Sex == "Male", ]
Female_Data <- HRData_Clean[HRData_Clean$Sex == "Female", ]

# Non-parametric test: Kruskal-Wallis for Males
male_kruskal <- kruskal.test(Change_from_Baseline ~ Stimulus, data = Male_Data)
cat("\n=== Kruskal-Wallis Test for Males ===\n")
print(male_kruskal)

# Non-parametric test: Kruskal-Wallis for Females
female_kruskal <- kruskal.test(Change_from_Baseline ~ Stimulus, data = Female_Data)
cat("\n=== Kruskal-Wallis Test for Females ===\n")
print(female_kruskal)

# Post-hoc tests if significant (Dunn's test for non-parametric)
if (male_kruskal$p.value < 0.05) {
  cat("\n=== Dunn's Post-hoc Test for Males ===\n")
  male_posthoc <- dunnTest(Change_from_Baseline ~ Stimulus, data = Male_Data, method = "bh")
  print(male_posthoc)
}

if (female_kruskal$p.value < 0.05) {
  cat("\n=== Dunn's Post-hoc Test for Females ===\n")
  female_posthoc <- dunnTest(Change_from_Baseline ~ Stimulus, data = Female_Data, method = "bh")
  print(female_posthoc)
}

# Calculate means for males and females
Male_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = Male_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Male_Means)[3] <- "Mean_Change_per_Bat"

Female_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Stimulus,
  data = Female_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Female_Means)[3] <- "Mean_Change_per_Bat"

# Plot for Males
male_plot <- ggplot(Male_Data, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA
  ) +
  geom_point(
    data = Male_Means,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  stat_compare_means(
    method = "kruskal.test",  # Use Kruskal-Wallis for comparison
    label.y = max(Male_Data$Change_from_Baseline, na.rm = TRUE) + 10
  ) +
  labs(
    title = "Males: Change in Heart Rate by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_manual(values = rep("#FFA500", 4))

# Plot for Females
female_plot <- ggplot(Female_Data, aes(x = Stimulus, y = Change_from_Baseline, fill = Stimulus)) +
  geom_boxplot(
    alpha = 0.7,
    outlier.shape = NA
  ) +
  geom_point(
    data = Female_Means,
    aes(x = Stimulus, y = Mean_Change_per_Bat),
    size = 3,
    alpha = 0.8,
    color = "black"
  ) +
  stat_compare_means(
    method = "kruskal.test",  # Use Kruskal-Wallis for comparison
    label.y = max(Female_Data$Change_from_Baseline, na.rm = TRUE) + 10
  ) +
  labs(
    title = "Females: Change in Heart Rate by Stimulus",
    x = "Stimulus Category",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_fill_manual(values = rep("#800080", 4))

# Display plots
print(male_plot)
print(female_plot)

# Combine plots
combined_plot <- ggarrange(
  male_plot,
  female_plot,
  ncol = 2,
  nrow = 1,
  labels = c("A", "B")
)
print(combined_plot)

# Summary statistics by sex and stimulus
summary_by_sex_stim <- aggregate(
  Change_from_Baseline ~ Sex + Stimulus,
  data = HRData_Clean,
  FUN = function(x) {
    c(n = length(x),
      Mean_Change = mean(x, na.rm = TRUE),
      SD_Change = sd(x, na.rm = TRUE),
      SE_Change = sd(x, na.rm = TRUE) / sqrt(length(x)))
  }
)
print(summary_by_sex_stim)



library(ggplot2)
library(ggpubr)
library(dplyr)

# Create grouped category variable
HRData_Clean$Category <- NA
HRData_Clean$Category[HRData_Clean$Stimulus %in% c("Aggression", "Distress")] <- "Social"
HRData_Clean$Category[HRData_Clean$Stimulus %in% c("Echolocation", "Distortion")] <- "Non-Social"

# Remove rows without a category
Category_Data <- HRData_Clean[!is.na(HRData_Clean$Category), ]

# Per-bat means per category
Category_Bat_Means <- aggregate(
  Change_from_Baseline ~ Bat_ID + Category,
  data = Category_Data,
  FUN = function(x) mean(x, na.rm = TRUE)
)
colnames(Category_Bat_Means)[3] <- "Mean_Change_per_Bat"

# Y-axis limit for sig bar
ymax <- max(Category_Data$Change_from_Baseline, na.rm = TRUE)


# Linear Mixed Effect Trial 
library(ggplot2)
library(ggpubr)
library(dplyr)
library(lme4)
library(lmerTest) 
# ── Subset: exclude Distortion ────────────────────────────────────────────────
SocNonSoc_Data <- HRData_Clean[
  HRData_Clean$Stimulus %in% c("Aggression", "Distress", "Echolocation"), 
]

SocNonSoc_Data$Category <- ifelse(
  SocNonSoc_Data$Stimulus %in% c("Aggression", "Distress"), "Social", "Non-Social"
)

# ── CRITICAL FIX: collapse to one median per bat per category FIRST ───────────
# This averages Aggression + Distress into a single Social value per bat
# so each bat contributes exactly ONE observation per category
Bat_Medians <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex + Category,
  data = SocNonSoc_Data,
  FUN  = function(x) median(x, na.rm = TRUE)
)
colnames(Bat_Medians)[4] <- "Median_Change"

# Verify: each bat should appear exactly twice (once Social, once Non-Social)
cat("\n=== Observations per bat (should be 2 each) ===\n")
print(table(Bat_Medians$Bat_ID))

# Set reference levels
Bat_Medians$Category <- factor(Bat_Medians$Category, levels = c("Non-Social", "Social"))
Bat_Medians$Sex      <- factor(Bat_Medians$Sex,      levels = c("Female", "Male"))

# ── LME on per-bat medians (balanced: 1 obs per bat per category) ─────────────
lme_model <- lmer(
  Median_Change ~ Category * Sex + (1 | Bat_ID),
  data = Bat_Medians
)

cat("\n=== Linear Mixed Effects Model ===\n")
print(summary(lme_model))

cat("\n=== ANOVA table (Type III) — interaction test ===\n")
print(anova(lme_model))

# Extract interaction p-value
lme_anova     <- anova(lme_model)
interaction_p <- lme_anova["Category:Sex", "Pr(>F)"]
int_p_label   <- ifelse(interaction_p < 0.001, "p < 0.001",
                        sprintf("p = %.3f", interaction_p))
int_sig       <- ifelse(interaction_p < 0.001, "***",
                        ifelse(interaction_p < 0.01,  "**",
                               ifelse(interaction_p < 0.05,  "*", "ns")))

cat(sprintf("\nSex × Category interaction: %s (%s)\n", int_sig, int_p_label))

# ── Group medians and SE for plots ────────────────────────────────────────────
# ── Group medians and IQR for plots ───────────────────────────────────────────
Group_Stats <- aggregate(
  Median_Change ~ Sex + Category,
  data = Bat_Medians,
  FUN  = function(x) c(
    Median = median(x),
    Q1     = unname(quantile(x, 0.25)),
    Q3     = unname(quantile(x, 0.75))
  )
)
Group_Stats <- do.call(data.frame, Group_Stats)
colnames(Group_Stats) <- c("Sex", "Category", "Median", "Q1", "Q3")
Group_Stats$Category  <- factor(Group_Stats$Category, levels = c("Non-Social", "Social"))

# ── Sex difference per category for bar plot ──────────────────────────────────
male_med   <- Group_Stats[Group_Stats$Sex == "Male",   c("Category", "Median")]
female_med <- Group_Stats[Group_Stats$Sex == "Female", c("Category", "Median")]
colnames(male_med)[2]   <- "Male_Median"
colnames(female_med)[2] <- "Female_Median"
Sex_Diff <- merge(male_med, female_med, by = "Category")
Sex_Diff$Sex_Difference <- abs(Sex_Diff$Female_Median - Sex_Diff$Male_Median)
Sex_Diff$Category <- factor(Sex_Diff$Category, levels = c("Non-Social", "Social"))

# ── Shared theme ──────────────────────────────────────────────────────────────
shared_theme <- theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10)
  )

sex_colors <- c("Male" = "#FFA500", "Female" = "#800080")

ymax_raw   <- max(Bat_Medians$Median_Change, na.rm = TRUE)
# ── Plot 1: Interaction plot ──────────────────────────────────────────────────


p1 <- ggplot(Group_Stats,
             aes(x = Category, y = Median, color = Sex, group = Sex)) +
  geom_line(
    data  = Bat_Medians,
    aes(x = Category, y = Median_Change, group = Bat_ID, color = Sex),
    alpha = 0.3, linewidth = 0.6
  ) +
  geom_point(
    data  = Bat_Medians,
    aes(x = Category, y = Median_Change, color = Sex),
    alpha = 0.7, size = 2
  ) +
  geom_line(linewidth = 1.4) +
  geom_point(size = 4) +
  geom_errorbar(
    aes(ymin = Q1, ymax = Q3),
    width = 0.08, linewidth = 1
  ) +

  labs(
    title = "",
    x     = "Stimulus Category",
    y     = "Δ Heart Rate (BPM)",
    color = "Sex"
  ) +
  scale_color_manual(values = sex_colors) +
  shared_theme +
  theme(legend.position = "none") +
  coord_cartesian(ylim = c(NA, ymax_raw * 1.12), clip = "off")

p1 <- p1 +
  annotate("text",
           x = 1.5,
           y = ymax_raw * 1.08,
           label = paste0(int_sig),
           size = 4.5, fontface = "bold")

# ── Plot 2: Side-by-side boxplots ─────────────────────────────────────────────
p2 <- ggplot(Bat_Medians,
             aes(x = Category, y = Median_Change, fill = Sex)) +
  geom_boxplot(
    width         = 0.5,
    position      = position_dodge(width = 0.7),
    outlier.shape = NA,
    alpha         = 0.7
  ) +
  geom_point(
    aes(color = Sex),
    position = position_dodge(width = 0.7),
    size = 3, alpha = 0.9
  ) +
  labs(
    title = "",
    x     = "Stimulus Category",
    y     = "Δ Heart Rate (BPM)",
    fill  = "Sex",
    color = "Sex"
  ) +
  scale_fill_manual(values  = sex_colors) +
  scale_color_manual(values = sex_colors) +
  shared_theme +
  theme(legend.position = "none") +
  coord_cartesian(clip = "off")

# ── Print & combine ───────────────────────────────────────────────────────────
print(p1)
print(p2)


combined <- ggarrange(p1, p2, ncol = 2, nrow = 1, labels = c("A", "B"))
print(combined)

print(Group_Stats)

print(anova(lme_model))



# ── Aggression vs Distortion: overall and Sex × Stimulus ──────────────────────
# Run after HRData_Clean is built (and Sex assigned).
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)   # install.packages("emmeans") if needed

# 1. One value per bat per stimulus (each bat hears BOTH stimuli -> paired design)
AD <- HRData_Clean %>%
  filter(Stimulus %in% c("Aggression", "Distortion")) %>%
  group_by(Bat_ID, Sex, Stimulus) %>%
  summarise(Change = median(Change_from_Baseline, na.rm = TRUE), .groups = "drop") %>%
  mutate(Stimulus = factor(Stimulus, levels = c("Distortion", "Aggression")),
         Sex      = factor(Sex,      levels = c("Female", "Male")))

# Keep only bats that have both stimuli (pairing needs complete pairs)
complete_bats <- AD %>% count(Bat_ID) %>% filter(n == 2) %>% pull(Bat_ID)
cat("Bats dropped for missing a stimulus:",
    setdiff(unique(AD$Bat_ID), complete_bats), "\n")
AD <- AD %>% filter(Bat_ID %in% complete_bats)
print(table(AD$Sex) / 2)   # bats per sex

# Wide format: per-bat difference (Aggression - Distortion)
AD_wide <- AD %>%
  pivot_wider(names_from = Stimulus, values_from = Change) %>%
  mutate(Diff = Aggression - Distortion)
print(AD_wide)

# 2. Nonparametric tests
# (a) Overall, ignoring sex: Wilcoxon signed-rank (paired)
cat("\n=== Aggression vs Distortion, paired (all bats) ===\n")
print(wilcox.test(AD_wide$Aggression, AD_wide$Distortion, paired = TRUE))

# (b) Sex × Stimulus interaction: does the Aggression-Distortion difference
#     differ between sexes?  Rank-sum test on per-bat differences
cat("\n=== Interaction: difference (Agg - Dist) by Sex ===\n")
print(wilcox.test(Diff ~ Sex, data = AD_wide))

# (c) Within each sex (paired), optional follow-up
for (s in levels(AD_wide$Sex)) {
  d <- AD_wide[AD_wide$Sex == s, ]
  cat("\n=== Paired test within", s, "===\n")
  print(wilcox.test(d$Aggression, d$Distortion, paired = TRUE))
}

# 3. Linear mixed model (same structure as your Social/Non-Social model)
#    Sum contrasts so the Stimulus main effect is averaged over sex
lme_AD <- lmer(Change ~ Stimulus * Sex + (1 | Bat_ID),
               data = AD,
               contrasts = list(Stimulus = "contr.sum", Sex = "contr.sum"))
cat("\n=== LMM ANOVA (Type III) ===\n")
print(anova(lme_AD))

# Stimulus effect averaged over sex (the "independent of sex" comparison)
cat("\n=== Aggression vs Distortion, averaged over sex ===\n")
print(emmeans(lme_AD, pairwise ~ Stimulus))

# Stimulus effect within each sex
cat("\n=== Aggression vs Distortion within each sex ===\n")
print(emmeans(lme_AD, pairwise ~ Stimulus | Sex))

# Check residuals (n is small, so look at these before trusting the LMM)
qqnorm(resid(lme_AD)); qqline(resid(lme_AD), col = "red")

# 4. Plot: one line per bat, group medians on top
sex_colors <- c("Male" = "#FFA500", "Female" = "#800080")

ggplot(AD, aes(x = Stimulus, y = Change, color = Sex)) +
  geom_line(aes(group = Bat_ID), alpha = 0.7, linewidth = 0.7) +
  geom_point(alpha = 0.6, size = 2) +
  stat_summary(aes(group = Sex), fun = median, geom = "line", linewidth = 1.4) +
  stat_summary(aes(group = Sex), fun = median, geom = "point", size = 4) +
  facet_wrap(~ Sex) +
  scale_color_manual(values = sex_colors) +
  labs(x = "Stimulus", y = "Δ Heart Rate (BPM)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

stim_colors <- c("Distortion" = "#E9C46A",   # muted gold
                 "Aggression" = "#E76F51")   # burnt coral-orange

pooled_test <- wilcox.test(AD_wide$Aggression, AD_wide$Distortion, paired = TRUE)
p_lab <- ifelse(pooled_test$p.value < 0.001, "p < 0.001",
                sprintf("p = %.3f", pooled_test$p.value))
p_sig <- ifelse(pooled_test$p.value < 0.001, "***",
                ifelse(pooled_test$p.value < 0.01,  "**",
                       ifelse(pooled_test$p.value < 0.05,  "*", "ns")))

ymax_AD <- max(AD$Change, na.rm = TRUE)

ggplot(AD, aes(x = Stimulus, y = Change)) +
  geom_boxplot(aes(fill = Stimulus), width = 0.5, alpha = 0.6, outlier.shape = NA) +
  geom_line(aes(group = Bat_ID), color = "gray50", alpha = 0.5, linewidth = 0.6) +
  geom_point(size = 3, alpha = 0.8, color = "black") +
  annotate("segment", x = 1, xend = 2, y = ymax_AD * 1.08, yend = ymax_AD * 1.08,
           linewidth = 0.8) +
  annotate("text", x = 1.5, y = ymax_AD * 1.13,
           label = p_sig, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = stim_colors) +
  labs(x = "Stimulus", y = "Δ Heart Rate (BPM)",
       subtitle = sprintf("Wilcoxon signed-rank, n = %d bats", nrow(AD_wide))) +
  coord_cartesian(ylim = c(NA, ymax_AD * 1.18), clip = "off") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10))



sex_colors <- c("Male" = "#FFA500", "Female" = "#800080")
box_alpha  <- 0.7
# One median per bat per stimulus
Bat_Stim_Medians <- aggregate(
  Change_from_Baseline ~ Bat_ID + Sex + Stimulus,
  data = HRData_Clean,
  FUN  = function(x) median(x, na.rm = TRUE)
)
colnames(Bat_Stim_Medians)[4] <- "Median_Change"

# Check: should be 6 per sex per stimulus
print(table(Bat_Stim_Medians$Sex, Bat_Stim_Medians$Stimulus))

ggplot(Bat_Stim_Medians, aes(x = Stimulus, y = Median_Change, fill = Sex)) +
  geom_boxplot(
    width = 0.5,
    position = position_dodge(width = 0.8),
    outlier.shape = NA,
    alpha = box_alpha
  ) +
  geom_point(
    aes(color = Sex),
    position = position_dodge(width = 0.8),
    size = 3,
    alpha = 0.4
  ) +
  stat_compare_means(
    aes(group = Sex),
    method = "wilcox.test",
    label  = "p.signif",
    hide.ns = TRUE,         # shows the real p; switch back to "p.signif" for the final figure
    label.y = max(Bat_Stim_Medians$Median_Change, na.rm = TRUE) + 10
  ) +
  labs(
    title = "Change in Heart Rate by Sex and Stimulus",
    x = "Stimulus",
    y = "Δ Heart Rate (BPM)",
    fill = "Sex", color = "Sex"
  ) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  scale_fill_manual(values = sex_colors) +
  scale_color_manual(values = sex_colors)


p_AD <- ggplot(AD, aes(x = Stimulus, y = Change)) +
  geom_boxplot(aes(fill = Stimulus), width = 0.4, alpha = 0.6, outlier.shape = NA) +
  geom_line(aes(group = Bat_ID), color = "gray50", alpha = 0.5, linewidth = 0.6) +
  geom_point(size = 2.5, alpha = 0.8, color = "black") +
  annotate("segment", x = 1, xend = 2, y = ymax_AD * 1.08, yend = ymax_AD * 1.08,
           linewidth = 0.8) +
  annotate("text", x = 1.5, y = ymax_AD * 1.13,
           label = p_sig, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = stim_colors) +
  scale_x_discrete(expand = expansion(add = 0.5)) +   # pulls the two boxes closer together
  labs(x = "Stimulus", y = "Δ Heart Rate (BPM)",
       subtitle = sprintf("Wilcoxon signed-rank, n = %d bats", nrow(AD_wide))) +
  coord_cartesian(ylim = c(NA, ymax_AD * 1.18), clip = "off") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        aspect.ratio = 2,                              # height = 2 × width -> skinny
        plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 10))

print(p_AD)

ggsave("Aggression_vs_Distortion.png", p_AD, width = 3.5, height = 5.5, dpi = 300)


ymax_pb <- max(per_bat$Median_Change, na.rm = TRUE)

ggplot(per_bat, aes(x = Sex, y = Median_Change, fill = Sex)) +
  geom_boxplot(width = 0.3, alpha = 0.7, outlier.shape = NA) +
  geom_point(aes(color = Sex), size = 3, alpha = 0.7) +
  annotate("segment", x = 1, xend = 2, y = ymax_pb * 1.08, yend = ymax_pb * 1.08,
           linewidth = 0.8) +
  annotate("text", x = 1.5, y = ymax_pb * 1.13,
           label = p_star(overall_test$p.value), size = 4.5, fontface = "bold") +
  labs(title = "Overall Change in Heart Rate by Sex",
       x = "Sex", y = "Δ Heart Rate (BPM)") +
  coord_cartesian(ylim = c(NA, ymax_pb * 1.18), clip = "off") +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = sex_colors) +
  scale_color_manual(values = sex_colors) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "none")

ymax_hr <- max(HRData_Clean$Change_from_Baseline, na.rm = TRUE)

ggplot(HRData_Clean, aes(x = Sex, y = Change_from_Baseline, fill = Sex, width = 0.3)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_point(
    data = Bat_Medians_Overall_Sex,
    aes(x = Sex, y = Median_Change_per_Bat, color = Sex),
    size = 3,
    alpha = 0.7
  ) +
  annotate("segment", x = 1, xend = 2, y = ymax_hr + 5, yend = ymax_hr + 5,
           linewidth = 0.8) +
  annotate("text", x = 1.5, y = ymax_hr + 12,
           label = p_star(overall_test$p.value), size = 5, fontface = "bold") +
  labs(
    title = "Overall Change in Heart Rate by Sex",
    x = "Sex",
    y = "Δ Heart Rate (BPM)"
  ) +
  theme_minimal(base_size = 14) +
  scale_fill_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  scale_color_manual(values = c("Male" = "#FFA500", "Female" = "#800080")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))