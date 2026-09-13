#!/usr/bin/env Rscript
# AlpacaVision - ALL publication data figures in ggplot2 (real data).
suppressPackageStartupMessages({
  library(ggplot2); library(jsonlite); library(scales); library(patchwork)
})
outdir <- file.path(normalizePath("."), "paper", "figures_R")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

ok <- c(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
        grey="#8A97A8", navy="#0B2545", sky="#56B4E9")
theme_pub <- function(base = 10) theme_minimal(base_size = base) +
  theme(panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        plot.title = element_text(face = "bold", size = base + 1),
        plot.subtitle = element_text(size = base - 1.5, colour = "grey30"),
        axis.title = element_text(face = "bold"),
        legend.position = "top", legend.title = element_blank(),
        legend.key.size = unit(0.9, "lines"))
sp <- function(p, n, w, h) {
  ggsave(file.path(outdir, n), p, width = w, height = h, device = cairo_pdf)
  ggsave(file.path(outdir, sub("pdf$", "png", n)), p, width = w, height = h, dpi = 160)
  cat("  ", n, "\n")
}

## ===== DATA =========================================================
n  <- fromJSON("outputs/figures/detector_clean_n_test_metrics.json")
s  <- fromJSON("outputs/figures/detector_clean_s_test_metrics.json")
cl <- fromJSON("outputs/figures/classifier_eyes_honest_metrics.json")
pr <- fromJSON("outputs/figures/classifier_eyes_honest_predictions.json")
cv <- read.csv("runs/detect/outputs/training_runs/stage1_clean_detector/results.csv",
               check.names = FALSE)

## ===== 1. DATASET (3 panels: funnel | splits | crops) ===============
d_funnel <- data.frame(stage = factor(c("Raw files","Duplicates","Unique"),
                        levels = c("Raw files","Duplicates","Unique")),
                        n = c(3088, 1037, 2051),
                        kind = c("total","removed","kept"))
pA <- ggplot(d_funnel, aes(stage, n, fill = kind)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = comma(n)), vjust = -0.4, size = 2.9, fontface = "bold") +
  scale_fill_manual(values = c(total=ok[["grey"]], removed=ok[["red"]], kept=ok[["green"]]), guide="none") +
  scale_y_continuous(expand = expansion(c(0, 0.14)), labels = comma) +
  labs(title = "(A) MD5 deduplication", x = NULL, y = "Images") + theme_pub()

d_split <- data.frame(split = factor(c("Train","Val","Test"), levels=c("Train","Val","Test")),
                      n = c(1435, 308, 308))
pB <- ggplot(d_split, aes(split, n, fill = split)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(comma(n), "\n", round(100*n/2051), "%")), vjust = -0.2, size = 2.6, lineheight = 0.85) +
  scale_fill_manual(values = c(ok[["blue"]], ok[["sky"]], ok[["orange"]]), guide="none") +
  scale_y_continuous(expand = expansion(c(0, 0.30)), labels = comma) +
  labs(title = "(B) Leakage-free split", x = NULL, y = NULL) + theme_pub()

d_crop <- data.frame(region = rep(c("Eyes","Legs"), each = 2),
                     cls = rep(c("Normal","Anomaly"), 2), n = c(371, 91, 443, 3))
pC <- ggplot(d_crop, aes(region, n, fill = cls)) +
  geom_col(position = position_dodge(0.7), width = 0.6) +
  geom_text(aes(label = n), position = position_dodge(0.7), vjust = -0.35, size = 2.7) +
  scale_fill_manual(values = c(Normal=ok[["green"]], Anomaly=ok[["red"]])) +
  scale_y_continuous(expand = expansion(c(0, 0.14))) +
  labs(title = "(C) Real ocular crops", x = NULL, y = NULL) + theme_pub()

fig_dataset <- pA + pB + pC + plot_layout(widths = c(1.1, 1, 1.1))
sp(fig_dataset, "fig_dataset.pdf", 7.0, 2.7)

## ===== 2. DETECTOR (metrics | training curves) ======================
det <- data.frame(metric = factor(rep(c("mAP@0.5","mAP@0.5:0.95","Precision","Recall"), 2),
                  levels = c("mAP@0.5","mAP@0.5:0.95","Precision","Recall")),
                  model = rep(c("YOLOv11n (2.58M)","YOLOv11s (9.43M)"), each = 4),
                  value = c(n$mAP50,n$mAP50_95,n$precision,n$recall,
                            s$mAP50,s$mAP50_95,s$precision,s$recall))
pM <- ggplot(det, aes(metric, value, fill = model)) +
  geom_col(position = position_dodge(0.7), width = 0.62) +
  geom_text(aes(label = sprintf("%.3f", value)), position = position_dodge(0.7),
            vjust = -0.35, size = 2.6) +
  scale_fill_manual(values = c(ok[["blue"]], ok[["orange"]])) +
  scale_y_continuous(limits = c(0,1), expand = expansion(c(0, 0.08))) +
  labs(title = "(A) Test-set metrics (n = 308)", x = NULL, y = "Score") + theme_pub()

trc <- data.frame(epoch = rep(cv$epoch, 2),
                  value = c(cv[["metrics/mAP50(B)"]], cv[["metrics/mAP50-95(B)"]]),
                  metric = rep(c("mAP@0.5","mAP@0.5:0.95"), each = nrow(cv)))
pT <- ggplot(trc, aes(epoch, value, colour = metric)) +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = c(ok[["blue"]], ok[["green"]])) +
  scale_y_continuous(limits = c(0,1)) +
  labs(title = "(B) YOLOv11n training", x = "Epoch", y = "Validation mAP") + theme_pub()

fig_detector <- pM + pT + plot_layout(widths = c(1.5, 1))
sp(fig_detector, "fig_detector.pdf", 7.0, 2.9)

## ===== 3. CLASSIFIER (confusion | ROC | PR) =========================
y <- pr$y_true_anomaly; sc <- pr$p_anomaly; P <- sum(y==1); N <- sum(y==0)
thr <- sort(unique(c(Inf, sc)), decreasing = TRUE)
roc <- data.frame(fpr = sapply(thr, function(t) sum(sc>=t & y==0)/N),
                  tpr = sapply(thr, function(t) sum(sc>=t & y==1)/P))
roc <- rbind(c(0,0), roc[order(roc$fpr, roc$tpr), ])
prc <- data.frame(recall = sapply(thr, function(t) sum(sc>=t & y==1)/P),
                  precision = sapply(thr, function(t){tp<-sum(sc>=t&y==1);fp<-sum(sc>=t&y==0);if(tp+fp==0)1 else tp/(tp+fp)}))

cm <- cl$confusion_matrix
cmdf <- expand.grid(True = c("Normal","Anomaly"), Pred = c("Normal","Anomaly"))
cmdf$True <- factor(cmdf$True, levels = c("Anomaly","Normal"))
cmdf$Pred <- factor(cmdf$Pred, levels = c("Normal","Anomaly"))
cmdf$n <- c(cm[1,1], cm[2,1], cm[1,2], cm[2,2])
pCM <- ggplot(cmdf, aes(Pred, True, fill = n)) +
  geom_tile(colour = "white", linewidth = 1.2) +
  geom_text(aes(label = n), size = 4.2, fontface = "bold") +
  scale_fill_gradient(low = "#EAF2FB", high = ok[["blue"]], guide = "none") +
  labs(title = "(A) Confusion matrix", x = "Predicted", y = "True") +
  theme_pub() + theme(panel.grid = element_blank())

pROC <- ggplot(roc, aes(fpr, tpr)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = ok[["grey"]]) +
  geom_line(colour = ok[["blue"]], linewidth = 0.9) +
  annotate("text", x = 0.62, y = 0.12, size = 2.7,
           label = sprintf("AUC = %.3f\n95%% CI [%.3f, %.3f]",
                           cl$auc_roc, cl$auc_roc_ci_95[1], cl$auc_roc_ci_95[2])) +
  coord_equal() +
  labs(title = "(B) ROC (at chance)", x = "False positive rate", y = "True positive rate") + theme_pub()

pPR <- ggplot(prc, aes(recall, precision)) +
  geom_hline(yintercept = P/(P+N), linetype = "dashed", colour = ok[["grey"]]) +
  geom_line(colour = ok[["orange"]], linewidth = 0.9) +
  annotate("text", x = 0.5, y = 0.85, size = 2.7, label = sprintf("AP = %.3f", cl$average_precision)) +
  scale_y_continuous(limits = c(0,1)) +
  labs(title = "(C) Precision-Recall", x = "Recall", y = "Precision") + theme_pub()

fig_classifier <- pCM + pROC + pPR + plot_layout(widths = c(1, 1.1, 1.1))
sp(fig_classifier, "fig_classifier.pdf", 7.0, 2.6)

## ===== 4. LEAKAGE by dedup stage (the thesis visual) ================
stages <- c("Raw (exact duplicates present)",
            "After cryptographic dedup",
            "After perceptual dedup")
leak <- data.frame(
  metric = factor(c(rep("Detector mAP@0.5", 3), rep("Classifier AUC-ROC", 2)),
                  levels = c("Detector mAP@0.5", "Classifier AUC-ROC")),
  stage  = factor(c(stages, stages[1:2]), levels = stages),
  value  = c(0.913, 0.860, 0.762, 0.824, 0.506))
figL <- ggplot(leak, aes(metric, value, fill = stage)) +
  geom_col(position = position_dodge2(width = 0.7, preserve = "single"), width = 0.62) +
  geom_text(aes(label = sprintf("%.3f", value)),
            position = position_dodge2(width = 0.7, preserve = "single"),
            vjust = -0.35, size = 2.9) +
  scale_fill_manual(values = unname(ok[c("red","orange","green")])) +
  scale_y_continuous(limits = c(0,1), expand = expansion(c(0, 0.12))) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(title = "Effect of data leakage on reported metrics",
       subtitle = "Cryptographic deduplication removes only part of the inflation",
       x = NULL, y = "Score") + theme_pub()
sp(figL, "fig_leakage.pdf", 4.6, 3.4)

cat("DONE - all ggplot2 figures in", outdir, "\n")
