########################################
# 0. Packages
########################################

# Install if needed:
# install.packages(c("tidyverse", "lme4", "lmerTest"))

library(tidyverse)
library(lme4)
library(lmerTest)   # gives p-values for lmer


########################################
# 1. Load CSVs
########################################

# Adjust paths if they live in a subfolder like "data/"
retina   <- read.csv("retina_regnet.csv", stringsAsFactors = FALSE)
eyeliner <- read.csv("eyeliner.csv",      stringsAsFactors = FALSE)
geoformer <- read.csv("geoformer.csv",   stringsAsFactors = FALSE)

# Standardise column names
retina <- retina %>%
  rename(
    image_id   = image_name,
    retina_mle = md_after
  )

eyeliner <- eyeliner %>%
  rename(
    image_id     = image_name,
    eyeliner_mle = md_after
  )

geoformer <- geoformer %>%
  rename(
    image_id      = gt_file,
    geoformer_mle = mle_distance
  )

# Merge into wide format: one row per image, one column per method
df_wide <- retina %>%
  inner_join(eyeliner, by = "image_id") %>%
  inner_join(geoformer, by = "image_id")

cat("Wide data shape:", nrow(df_wide), "rows x", ncol(df_wide), "cols\n")
head(df_wide)


########################################
# 2. Add class label (S / A / P from first character)
########################################

df_wide <- df_wide %>%
  mutate(
    cls = substr(image_id, 1, 1)
  )

cat("\nClass counts:\n")
print(table(df_wide$cls))


########################################
# 3. Long-format table for modelling
########################################

df_long <- df_wide %>%
  pivot_longer(
    cols = c(retina_mle, eyeliner_mle, geoformer_mle),
    names_to = "method",
    values_to = "mle"
  ) %>%
  mutate(
    method = factor(method),
    cls    = factor(cls),
    image_id = factor(image_id)
  )

cat("\nLong data shape:", nrow(df_long), "rows x", ncol(df_long), "cols\n")
head(df_long)


########################################
# 4. Overall Friedman test (all images)
########################################
# Nonparametric repeated-measures across 3 methods.

m_mat <- as.matrix(df_wide[, c("retina_mle", "eyeliner_mle", "geoformer_mle")])

friedman_overall <- friedman.test(m_mat)

cat("\n=== Overall Friedman test (all images) ===\n")
print(friedman_overall)


########################################
# 5. Overall pairwise Wilcoxon (paired)
#    with Bonferroni correction
########################################

pairs <- list(
  c("retina_mle",   "eyeliner_mle"),
  c("retina_mle",   "geoformer_mle"),
  c("eyeliner_mle", "geoformer_mle")
)
m <- length(pairs)  # for Bonferroni

cat("\n=== Overall pairwise Wilcoxon signed-rank tests (all images) ===\n")

for (p in pairs) {
  a <- p[1]
  b <- p[2]
  test <- wilcox.test(df_wide[[a]], df_wide[[b]],
                      paired = TRUE, exact = FALSE)
  p_raw <- test$p.value
  p_adj <- min(p_raw * m, 1)
  diff_mean <- mean(df_wide[[a]] - df_wide[[b]], na.rm = TRUE)
  
  cat(sprintf(
    "%s vs %s: V = %.1f, p_raw = %.3e, p_Bonf = %.3e, mean diff = %.4f\n",
    a, b, test$statistic, p_raw, p_adj, diff_mean
  ))
}


########################################
# 6. Per-class Friedman tests (S, A, P)
########################################

cat("\n=== Per-class Friedman tests ===\n")

for (cls_val in sort(unique(df_wide$cls))) {
  sub_df <- df_wide %>% filter(cls == cls_val)
  m_sub <- as.matrix(sub_df[, c("retina_mle", "eyeliner_mle", "geoformer_mle")])
  
  if (nrow(sub_df) > 0) {
    fr <- friedman.test(m_sub)
    cat(sprintf(
      "Class %s: n = %d, Chi-square = %.3f, p = %.3e\n",
      cls_val, nrow(sub_df), fr$statistic, fr$p.value
    ))
  }
}


########################################
# 7. Per-class pairwise Wilcoxon tests
########################################

cat("\n=== Per-class pairwise Wilcoxon tests ===\n")

for (cls_val in sort(unique(df_wide$cls))) {
  sub_df <- df_wide %>% filter(cls == cls_val)
  cat(sprintf("\nClass %s (n = %d)\n", cls_val, nrow(sub_df)))
  
  if (nrow(sub_df) > 0) {
    for (p in pairs) {
      a <- p[1]
      b <- p[2]
      
      test <- wilcox.test(sub_df[[a]], sub_df[[b]],
                          paired = TRUE, exact = FALSE)
      p_raw <- test$p.value
      p_adj <- min(p_raw * m, 1)
      diff_mean <- mean(sub_df[[a]] - sub_df[[b]], na.rm = TRUE)
      
      cat(sprintf(
        "  %s vs %s: V = %.1f, p_raw = %.3e, p_Bonf = %.3e, mean diff = %.4f\n",
        a, b, test$statistic, p_raw, p_adj, diff_mean
      ))
    }
  }
}


########################################
# 8. OPTIONAL: Linear mixed-effects model
#    mle ~ method * cls + (1 | image_id)
########################################

cat("\n=== Mixed-Effects Model: mle ~ method * cls + (1 | image_id) ===\n")

# Use df_long
lmm <- lmer(
  mle ~ method * cls + (1 | image_id),
  data = df_long
)

summary(lmm)

# If you want ANOVA-style tests for fixed effects:
cat("\nType III ANOVA for fixed effects:\n")
print(anova(lmm, type = 3))



# visually checking normaility of MLE residuals
par(mfrow=c(1,2))
qqnorm(residuals(lmm))
qqline(residuals(lmm))
hist(residuals(lmm), breaks=20, main="Histogram of residuals", xlab="Residuals")
par(mfrow=c(1,1))
# End of script


ben_shapiro =shapiro.test(residuals(lmm))  # test for normality of residuals
print(ben_shapiro)
# If p < 0.05, residuals are not normally distributed


# mean and standard error
summary(df_long %>%
          group_by(method) %>%
          summarise(
            mean_mle = mean(mle, na.rm = TRUE),
            se_mle   = sd(mle, na.rm = TRUE) / sqrt(n())
          ))

# save m_mat to csv

write.csv(m_mat, "mle_matrix.csv", row.names = FALSE)


# summary mle matrix
summary(m_mat)
# mean and standard error per method
apply(m_mat, 2, function(x) c(mean = mean(x, na.rm = TRUE),
                              se   = sd(x, na.rm = TRUE) / sqrt(length(na.omit(x)))))

# remove mfrow
par(mfrow=c(1,1))
# boxplot of mle per method
boxplot(m_mat,
        main = "MLE per Method",
        xlab = "Method",
        ylab = "MLE",
        names = c("Retina", "Eyeliner", "Geoformer"))


# save output of this script to a text file
sink("analysis_output.txt")
