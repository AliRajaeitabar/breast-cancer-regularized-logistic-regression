# =========================================================
# Cancer Classification Using Regularized Logistic Regression
# A Comparison of Lasso, Ridge, and Elastic Net under Multicollinearity
# =========================================================

# =========================
# 1. Load required packages
# =========================

# tidyverse: data manipulation and visualization
library(tidyverse)

# caTools: train/test split
library(caTools)

# ROCR: ROC curve and threshold analysis
library(ROCR)

# rio: data import
library(rio)

# glmnet: regularized logistic regression
library(glmnet)

# pROC: AUC calculation and ROC comparison
library(pROC)

# here: handle file paths relative to the project root
library(here)

# car: provides VIF for multicollinearity diagnostics
library(car)

# =========================
# 2. Import dataset
# =========================

# Load dataset using a relative path for reproducibility
data <- import(here("data", "data.csv"))

# Inspect dataset structure and first rows
str(data)
head(data)


# =========================
# 3. Initial data inspection and cleaning
# =========================

# Remove the first column because it contains ID values
# and does not provide predictive information
data <- data[, -1]

# Remove the last column because it contains only missing values
data <- data[, -ncol(data)]

# Check updated dimensions
dim(data)

# Convert character columns to factors for modeling
data <- data %>%
  mutate(across(where(is.character), as.factor))

# Verify structure after conversion
str(data)

# Remove duplicate rows if any exist

data <- data %>% distinct()
# =========================
# 4. Data quality notes
# =========================

# Missing values were reviewed and no missing values were detected
colSums(is.na(data))
colMeans(is.na(data))

# All variables were reviewed, and no numeric columns were found to represent
# categorical labels encoded as numbers. Therefore, no additional factor
# conversion is required beyond character columns.

# The dataset does not contain date or time variables,
# so no time-based preprocessing is required.

# No manual standardization is applied before the baseline logistic regression.
# For regularized models, glmnet standardizes predictors by default.

# Based on histogram inspection, no impossible or invalid values were detected.
# Some variables show potential extreme values, which may reflect skewness
# rather than true outliers. No observations are removed at this stage.


# =========================
# 5. Exploratory analysis
# =========================

# Reshape numeric variables for visualization
numeric_data <- data %>%
  select(where(is.numeric)) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "value"
  )

# Plot histograms for all numeric variables
ggplot(numeric_data, aes(x = value)) +
  geom_histogram(bins = 30) +
  facet_wrap(~ variable, scales = "free") +
  labs(
    title = "Distribution of Numeric Variables",
    x = "Value",
    y = "Frequency"
  )


# =========================
# 6. Train-test split
# =========================

# Set seed for reproducibility
set.seed(123)

# Split data into training and testing sets
split <- sample.split(data$diagnosis, SplitRatio = 0.8)

training <- subset(data, split == TRUE)
testing  <- subset(data, split == FALSE)

# =========================
# 7. Baseline logistic regression
# =========================

# Fit baseline logistic regression model
baseline_model <- glm(
  diagnosis ~ .,
  data = training,
  family = binomial
)

# Review baseline model summary
summary(baseline_model)

# Note:
# If warnings such as "algorithm did not converge" or
# "fitted probabilities numerically 0 or 1 occurred" appear,
# this suggests separation and/or multicollinearity issues.
# These problems motivate the use of regularized models.
# High VIF values indicate potential multicollinearity among predictors
vif(baseline_model)

# Predicted probabilities on test data
prob_baseline <- predict(
  baseline_model,
  newdata = testing,
  type = "response"
)

# Convert probabilities to class labels
pred_baseline_class <- factor(
  ifelse(prob_baseline > 0.5, "M", "B"),
  levels = levels(testing$diagnosis)
)

# Confusion matrix
cm_baseline <- table(
  Actual = testing$diagnosis,
  Predicted = pred_baseline_class
)
cm_baseline
# Accuracy
accuracy_baseline <- sum(diag(cm_baseline)) / sum(cm_baseline)
accuracy_baseline

# =========================
# 8. Prepare data for glmnet
# =========================

# Create model matrices for regularized models
x_train <- model.matrix(diagnosis ~ ., data = training)[, -1]
x_test  <- model.matrix(diagnosis ~ ., data = testing)[, -1]

# Convert target variable to numeric format required by glmnet
# 1 = malignant, 0 = benign
y_train <- ifelse(training$diagnosis == "M", 1, 0)
y_test  <- ifelse(testing$diagnosis == "M", 1, 0)


# =========================
# 9. Lasso logistic regression
# =========================

# Cross-validated Lasso to find optimal lambda
cv_lasso <- cv.glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 1
)

# Fit final Lasso model using optimal lambda
lasso_model <- glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 1,
  lambda = cv_lasso$lambda.min
)

# Predicted probabilities on test data
prob_lasso <- predict(lasso_model, newx = x_test, type = "response")
prob_lasso <- as.vector(prob_lasso)

# Convert probabilities to predicted classes
pred_lasso <- prob_lasso > 0.5

# Confusion matrix
cm_lasso <- table(
  Actual = y_test,
  Predicted = pred_lasso
)
cm_lasso

# Extract coefficients
lasso_coef <- coef(lasso_model)
lasso_coef

# =========================
# 10. Ridge logistic regression
# =========================

# Cross-validated Ridge to find optimal lambda
cv_ridge <- cv.glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 0
)

# Fit final Ridge model
ridge_model <- glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 0,
  lambda = cv_ridge$lambda.min
)

# Predicted probabilities
prob_ridge <- predict(ridge_model, newx = x_test, type = "response")
prob_ridge <- as.vector(prob_ridge)

pred_ridge <- prob_ridge > 0.5

# Confusion matrix
cm_ridge <- table(
  Actual = y_test,
  Predicted = pred_ridge
)

cm_ridge

# Extract coefficients
ridge_coef <- coef(ridge_model)
as.matrix(ridge_coef)

# =========================
# 11. Elastic Net logistic regression
# =========================

# Cross-validated Elastic Net to find optimal lambda
cv_enet <- cv.glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 0.5
)

# Fit final Elastic Net model
enet_model <- glmnet(
  x_train,
  y_train,
  family = "binomial",
  alpha = 0.5,
  lambda = cv_enet$lambda.min
)

# Predicted probabilities
prob_enet <- predict(enet_model, newx = x_test, type = "response")
prob_enet <- as.vector(prob_enet)

# Convert probabilities to predicted classes
pred_enet <- prob_enet > 0.5

# Confusion matrix
cm_enet <- table(
  Actual = y_test,
  Predicted = pred_enet
)
cm_enet

# Extract coefficients
enet_coef <- coef(enet_model)
as.matrix(enet_coef)

# =========================
# 12. AUC Calculation for Regularized Logistic Models
# =========================
# ROC objects using probabilities
auc_lasso <- roc(y_test, prob_lasso)
auc_ridge <- roc(y_test, prob_ridge)
auc_enet  <- roc(y_test, prob_enet)

# AUC values
auc(auc_lasso)
auc(auc_ridge)
auc(auc_enet)

# =========================
# 13. Comparison of ROC Curves for Regularized Logistic Models
# =========================

plot(
  auc_lasso,
  col = "red",
  lwd = 2,
  main = "ROC Curves for Regularized Logistic Models",
  legacy.axes = TRUE
)

plot(
  auc_ridge,
  add = TRUE,
  col = "blue",
  lwd = 2
)

plot(
  auc_enet,
  add = TRUE,
  col = "green",
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    paste("Lasso AUC =", round(auc(auc_lasso), 4)),
    paste("Ridge AUC =", round(auc(auc_ridge), 4)),
    paste("Elastic Net AUC =", round(auc(auc_enet), 4))
  ),
  col = c("red", "blue", "green"),
  lwd = 2
)
# =========================
# 14. Use ROC analysis to identify an optimal classification threshold for the Lasso model.
# =========================

# The goal is to reduce false negatives (missed malignant cases),
# which is critical in cancer classification.

ROCRpred_lasso <- prediction(prob_lasso, y_test)
ROCRperf_lasso <- performance(ROCRpred_lasso, "tpr", "fpr")

plot(
  ROCRperf_lasso,
  colorize = TRUE,
  print.cutoffs.at = seq(0, 1, by = 0.1),
  main = "ROC Curve - Lasso Logistic Regression"
)
abline(a = 0, b = 1, lty = 2)

# Evaluate Lasso model at a lower threshold to reduce false negatives,
# and compute key classification metrics from the confusion matrix.
# Apply chosen threshold (e.g., 0.25) to predicted probabilities
pred_lasso_class <- ifelse(prob_lasso > 0.25, 1, 0)

# Confusion matrix
cm <- table(
  Actual = y_test,
  Predicted = pred_lasso_class
)

cm

# Extract confusion matrix values
TN <- cm[1, 1]
FP <- cm[1, 2]
FN <- cm[2, 1]
TP <- cm[2, 2]

# Performance metrics
accuracy <- (TP + TN) / sum(cm)
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)              # Sensitivity
specificity <- TN / (TN + FP)
f1 <- 2 * (precision * recall) / (precision + recall)
fpr <- FP / (FP + TN)
balanced_acc <- (recall + specificity) / 2

# Combine results
results <- c(
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  specificity = specificity,
  f1 = f1,
  fpr = fpr,
  balanced_accuracy = balanced_acc
)

print(results)
