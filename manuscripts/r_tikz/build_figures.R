#!/usr/bin/env Rscript
# Rendering only: completed JSON reports -> R/grid -> editable TikZ.
# All dimensions below are millimetres at the final 180 mm journal width.
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(tikzDevice)
})
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
ROOT <- normalizePath(file.path(dirname(script), "../.."))
OUT <- file.path(ROOT, "manuscripts/figures")
TEX <- file.path(OUT, "tex")
DATA <- file.path(OUT, "source_data")
BUILD <- file.path(ROOT, "outputs/figure_r_tikz")
for (d in c(TEX, DATA, BUILD)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
requested <- commandArgs(trailingOnly = TRUE)

NAVY <- "#234E70"; BLUE <- "#4C78A8"; TEAL <- "#1F8A8A"
GREEN <- "#2E7D32"; ORANGE <- "#C46A1B"; PURPLE <- "#6B4C9A"
RED <- "#B44B4B"; GRAY <- "#73777B"; DARK <- "#263238"
ARMS <- c("Healer" = GREEN, "Non-healer" = ORANGE, "Healthy" = PURPLE)
font_commands <- c("\\usepackage{fontspec}", "\\setmainfont{Arial}", "\\setsansfont{Arial}")
options(tikzDefaultEngine = "xetex",
        tikzUnicodeMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricsDictionary = file.path(BUILD, "arial-metrics"))

theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  # Figure lettering is deliberately monochrome and bold. Data marks and
  # reference lines retain colour; every textual element remains black.
  text = element_text(colour = "black", face = "bold"),
  axis.text = element_text(size = 7.5, colour = "black", face = "bold"),
  axis.title.x = element_text(size = 8, colour = "black", face = "bold", margin = margin(t = 1.4, unit = "mm")),
  axis.title.y = element_text(size = 8, colour = "black", face = "bold", margin = margin(r = 1.3, unit = "mm")),
  axis.ticks = element_line(linewidth = 0.25, colour = GRAY),
  axis.ticks.length = unit(1, "mm"),
  axis.line = element_line(linewidth = 0.25, colour = GRAY),
  panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E2E7EB"),
  panel.grid.minor = element_blank(),
  plot.margin = margin(t = 1.8, r = 1.2, b = 1.2, l = 1.2, unit = "mm"),
  plot.background = element_rect(fill = "white", colour = NA),
  legend.position = "bottom",
  legend.text = element_text(size = 7, colour = "black", face = "bold"),
  legend.title = element_blank(),
  legend.key.size = unit(3.2, "mm"), legend.spacing.x = unit(1.2, "mm"),
  legend.margin = margin(0, 0, 0, 0, unit = "mm"),
  legend.box.spacing = unit(1.3, "mm")
))
sources <- list(); active_sources <- character(); figures <- list()
read_report <- function(path) {
  full <- file.path(ROOT, path)
  d <- fromJSON(full, simplifyVector = FALSE)
  if (identical(d$status, "failed")) stop("Failed source: ", path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  d
}
repaired <- function(name) read_report(paste0("outputs/analysis/", name, "/report.json"))
historical <- function(name) read_report(paste0("outputs/", name, "/report.json"))
num <- function(rows, key) vapply(rows, function(x) as.numeric(x[[key]]), numeric(1))
chr <- function(rows, key) vapply(rows, function(x) as.character(x[[key]]), character(1))
vector <- function(x) as.numeric(unlist(x, use.names = FALSE))
ordered_factor <- function(x) factor(x, levels = unique(x))
panel <- function(plot, title, data) list(plot = plot, title = title, data = data)
zero_h <- function() geom_hline(yintercept = 0, colour = GRAY, linewidth = .25)
zero_v <- function(x = 0) geom_vline(xintercept = x, colour = GRAY, linewidth = .3, linetype = "dashed")
xgrid <- function() theme(panel.grid.major.y = element_blank(),
                         panel.grid.major.x = element_line(linewidth = .18, colour = "#E2E7EB"))
columns <- function(d, ylabel, colors, limits = NULL) {
  d$label <- ordered_factor(d$label)
  p <- ggplot(d, aes(label, value, fill = label)) +
    geom_col(width = .58) + scale_fill_manual(values = colors) +
    labs(x = NULL, y = ylabel) + theme(legend.position = "none")
  if (!is.null(limits)) p <- p + coord_cartesian(ylim = limits)
  p
}
forest <- function(d, xlabel = "AUC", ref = .5, limits = c(0, 1)) {
  d$label <- factor(d$label, levels = rev(unique(d$label)))
  ggplot(d, aes(value, label)) + zero_v(ref) +
    geom_errorbar(aes(xmin = low, xmax = high), orientation = "y", width = .15,
                  colour = NAVY, linewidth = .55) +
    geom_point(size = 2, colour = NAVY) +
    scale_x_continuous(limits = limits, expand = expansion(mult = .025)) +
    labs(x = xlabel, y = NULL) + xgrid()
}
record_figure <- function(name, width, height, panels, heading_centres) {
  data_files <- character()
  for (i in seq_along(panels)) {
    path <- file.path(DATA, paste0(name, "_", LETTERS[i], ".csv"))
    write.csv(panels[[i]]$data, path, row.names = FALSE, na = "")
    data_files <- c(data_files, sub(paste0(ROOT, "/"), "", path, fixed = TRUE))
  }
  figures[[name]] <<- list(width_mm = width, height_mm = height,
      panels = vapply(panels, `[[`, character(1), "title"),
      headings = paste(LETTERS[seq_along(panels)], vapply(panels, `[[`, character(1), "title")),
      heading_centres_mm = heading_centres,
      sources = active_sources, data = data_files)
  active_sources <<- character()
  message("TikZ: ", name)
}
draw_figure <- function(name, panels, height = 82, widths = NULL, nrow = 1, gap = 5) {
  ncol <- length(panels) / nrow
  stopifnot(ncol == as.integer(ncol))
  width <- 180; left <- 3; top <- 2; bottom <- 1.5; row_gap <- 4
  if (is.null(widths)) widths <- rep((width - 2 * left - gap * (ncol - 1)) / ncol, ncol)
  stopifnot(abs(sum(widths) + gap * (ncol - 1) - 174) < .01)
  row_h <- (height - top - bottom - row_gap * (nrow - 1)) / nrow
  tikz(file.path(TEX, paste0(name, ".tex")), width = width / 25.4, height = height / 25.4,
       pointsize = 8, standAlone = TRUE, engine = "xetex", sanitize = TRUE,
       timestamp = FALSE, documentDeclaration = "\\documentclass[10pt]{standalone}",
       packages = c("\\usepackage{tikz}", font_commands))
  on.exit(dev.off())
  grid.newpage()
  for (i in seq_along(panels)) {
    col <- (i - 1) %% ncol + 1; row <- (i - 1) %/% ncol + 1
    x <- left + sum(head(widths, col - 1)) + gap * (col - 1)
    y <- height - top - (row - 1) * (row_h + row_gap)
    # One centred text object keeps the panel letter attached to its title.
    grid.text(paste(LETTERS[i], panels[[i]]$title),
              unit(x + widths[col] / 2, "mm"), unit(y, "mm"), just = c("centre", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 9, fontface = "bold", col = "black"))
    pushViewport(viewport(x = unit(x, "mm"), y = unit(y - 6.3, "mm"),
                          width = unit(widths[col], "mm"), height = unit(row_h - 6.3, "mm"),
                          just = c("left", "top"), clip = "off"))
    grid.draw(ggplotGrob(panels[[i]]$plot))
    popViewport()
  }
  centres <- left + c(0, head(cumsum(widths), -1)) + gap * (seq_len(ncol) - 1) + widths / 2
  record_figure(name, width, height, panels, rep(centres, nrow))
}
write_workflow <- function(name, height, body, steps, heading_centres) {
  colors <- paste0("\\definecolor{", c("navy", "blue", "teal", "orange", "purple", "green", "gray"),
                   "}{HTML}{", sub("#", "", c(NAVY, BLUE, TEAL, ORANGE, PURPLE, GREEN, GRAY)), "}")
  lines <- c("\\documentclass[10pt]{standalone}", "\\usepackage{tikz}", font_commands,
    "\\usetikzlibrary{arrows.meta,patterns}", colors, "\\begin{document}",
    "\\begin{tikzpicture}[x=1mm,y=1mm,font=\\fontsize{8}{10}\\selectfont\\bfseries,text=black,>=Latex]",
    sprintf("\\path[use as bounding box] (0,0) rectangle (180,%s);", height),
    "\\tikzset{step/.style={draw=#1!40,fill=#1!8,rounded corners=1mm,",
    "minimum height=13mm,text width=21mm,align=center,inner sep=1.5mm,text=black},",
    "link/.style={->,draw=gray,line width=0.45pt}}", body,
    "\\end{tikzpicture}", "\\end{document}")
  writeLines(lines, file.path(TEX, paste0(name, ".tex")))
  record_figure(name, 180, height,
                lapply(steps, function(s) panel(NULL, s, data.frame(step = s))), heading_centres)
}
methods <- c("topic_simplex_theta0", "module_score", "pca", "nmf")
method_names <- c("Fibroblast\ntopic", "Module", "PCA", "NMF")

figure1_workflow <- function() {
  body <- readLines(file.path(ROOT, "manuscripts/r_tikz/temporal_design.tikz"))
  write_workflow("figure1_workflow", 111, body,
    c("Missing time point", "New donor: example fold", "Common evaluation and complementary sensitivity analyses"), c(43, 135, 90))
}

figure2_wound7_transfer <- function() {
  h <- repaired("human_temporal_flow"); t <- repaired("donor_transfer_flow"); g <- repaired("donor_geometry_inference")
  a <- data.frame(label = c("Prediction", "Stay-still"), value = c(h$all_donors$predicted, h$all_donors$standstill))
  pa <- columns(a, "Energy distance", c(BLUE, GRAY), c(0, 1.7))
  b <- data.frame(label = names(t$per_donor), value = 100 * num(t$per_donor, "improvement"))
  pb <- columns(b, "Improvement (%)", c(GREEN, BLUE, TEAL), c(0, 100))
  c <- data.frame(group = c(rep("Same-time\ndonor", length(g$pairwise_inputs$between_donor_same_time)),
       rep("Same-donor\ntime", length(g$pairwise_inputs$between_time_same_donor))),
       value = c(num(g$pairwise_inputs$between_donor_same_time, "d"), num(g$pairwise_inputs$between_time_same_donor, "d")))
  c$group <- factor(c$group, levels = unique(c$group))
  pc <- ggplot(c, aes(group, value, colour = group, fill = group)) +
    geom_boxplot(width = .5, outlier.shape = NA, alpha = .15, linewidth = .4) +
    geom_point(position = position_jitter(width = .09, height = 0, seed = 47), size = 1.3, alpha = .8) +
    scale_colour_manual(values = c(TEAL, ORANGE)) + scale_fill_manual(values = c(TEAL, ORANGE)) +
    labs(x = NULL, y = "Pairwise distance") + theme(legend.position = "none")
  draw_figure("figure2_wound7_transfer", list(panel(pa, "Held-out Wound7", a),
    panel(pb, "Donor transfer", b), panel(pc, "Donor–time geometry", c)), height = 67)
}
figure3_time_axes <- function() {
  r <- repaired("time_parameterisation")
  keys <- c("rank", "days", "sqrt_days", "log_days")
  labels <- c("Rank", "Linear days", "Square-root days", "Log days")
  d <- do.call(rbind, lapply(seq_along(keys), function(i) data.frame(axis = labels[i], fraction = num(r$scans[[keys[i]]], "frac"), ed = num(r$scans[[keys[i]]], "ed"))))
  target <- data.frame(axis = labels, fraction = num(r$axes_results[keys], "claimed_fraction"), ed = num(r$axes_results[keys], "ed_at_claimed"))
  p <- ggplot(d, aes(fraction, ed, colour = axis, linetype = axis)) +
    geom_hline(yintercept = r$standstill, linetype = "dashed", colour = GRAY, linewidth = .35) +
    geom_line(linewidth = .65) + geom_point(data = target, size = 2) +
    scale_colour_manual(values = setNames(c(BLUE, RED, GREEN, PURPLE), labels)) +
    scale_linetype_manual(values = setNames(c("solid", "longdash", "dotted", "dotdash"), labels)) +
    labs(x = "Fraction along predicted path", y = "Energy distance") +
    guides(colour = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1))
  # Retain the complete observed y range; do not truncate the linear-days curve.
  exported <- rbind(transform(d, point_type = "scan"), transform(target, point_type = "held_out_target"),
    data.frame(axis = "Stay-still", fraction = NA, ed = r$standstill, point_type = "baseline"))
  draw_figure("figure3_time_axes", list(panel(p, "Time parameterization", exported)), height = 70)
}
figure4_methods_benchmark <- function() {
  r <- repaired("donor_conditioned_benchmark")
  keys <- c("M0_standstill", "M1_shared_cfm", "M2_mean_displacement", "M3_nearest_donor", "M4_conditioned_cfm", "M5_optimal_transport")
  a <- data.frame(label = c("Stay-\nstill", "Shared\nCFM", "Mean\nshift", "Nearest\ndonor", "Cond.\nCFM", "OT"), value = num(r$summary[keys], "mean_ed"))
  pa <- columns(a, "Mean energy distance", c(GRAY, NAVY, BLUE, TEAL, PURPLE, ORANGE))
  b <- do.call(rbind, lapply(names(r$in_sample), function(k) data.frame(fold = k,
    field = c("Shared", "Conditioned"), value = c(r$in_sample[[k]]$shared, r$in_sample[[k]]$conditioned))))
  b$field <- factor(b$field, levels = c("Shared", "Conditioned"))
  pb <- ggplot(b, aes(field, value, group = fold)) + geom_line(colour = PURPLE, alpha = .6, linewidth = .5) +
    geom_point(colour = PURPLE, size = 1.5) + labs(x = NULL, y = "Energy distance")
  draw_figure("figure4_methods_benchmark", list(panel(pa, "Held-out donor", a),
    panel(pb, "Training donors", b)), height = 68, widths = c(99, 70))
}
figure5_donor_curve_stability <- function() {
  s <- repaired("donor_curve_seed_stability"); c <- repaired("donor_count_curve"); r <- repaired("donor_curve_inference")
  a <- data.frame(k = rep(1:2, each = 3), seed = rep(names(s$per_seed), 2),
    value = c(num(s$per_seed, "k1_mean_ed"), num(s$per_seed, "k2_mean_ed")))
  fixed <- data.frame(k = 1:2, seed = "0", value = c(c$curve$k_1$shared_cfm_mean_ed, c$curve$k_2$shared_cfm_mean_ed))
  pa <- ggplot(a, aes(k, value)) + geom_line(data = fixed, colour = NAVY, linewidth = .6) +
    geom_point(aes(shape = seed), size = 2, colour = BLUE) +
    scale_shape_manual(values = c(16, 17, 15), name = "Seed") +
    scale_x_continuous(breaks = 1:2, labels = c("k = 1", "k = 2"), limits = c(.7, 2.3)) +
    labs(x = NULL, y = "Held-out energy distance") + theme(legend.title = element_text(size = 7, colour = "black", face = "bold"))
  b <- data.frame(seed = names(s$per_seed), reduction = vector(s$reduction_across_seeds$values),
    low = r$aggregate$donor_block_bootstrap_95_ci[[1]], high = r$aggregate$donor_block_bootstrap_95_ci[[2]],
    mean = r$aggregate$mean_reduction_across_donor_blocks)
  pb <- ggplot(b, aes(seed, reduction)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = b$low[1], ymax = b$high[1], fill = "#DFEAF1", colour = NA) +
    geom_hline(yintercept = 0, colour = GRAY, linewidth = .3) +
    geom_hline(yintercept = b$mean[1], colour = NAVY, linewidth = .55) +
    geom_point(size = 2.2, colour = ORANGE) + coord_cartesian(ylim = c(-.04, .39)) +
    labs(x = "Training seed", y = "ED reduction (k=1 − k=2)")
  draw_figure("figure5_donor_curve_stability", list(panel(pa, "Donor curve", a),
    panel(pb, "Seed stability", b)), height = 70)
}
figure6_mouse_arms <- function() {
  r <- repaired("mouse_all_arms_pod7_audit"); keys <- c("NDB", "PDB", "GDB")
  a <- data.frame(label = keys, value = 100 * num(r$arms[keys], "improvement_fraction"))
  pa <- columns(a, "Improvement (%)", c(GRAY, ORANGE, GREEN), c(-20, 65)) + zero_h()
  # Diagonal strokes drawn as vector segments within the off-path PDB bar.
  hatch <- function(center, half, top, spacing) {
    starts <- seq(-spacing, top, by = spacing)
    do.call(rbind, lapply(starts, function(y) {
      lo <- max(0, -y / spacing); hi <- min(1, (top - y) / spacing)
      if (lo >= hi) return(NULL)
      data.frame(x = center - half + 2 * half * lo, xend = center - half + 2 * half * hi,
                 y = y + spacing * lo, yend = y + spacing * hi)
    }))
  }
  pa <- pa + geom_segment(data = hatch(2, .29, a$value[2], 5),
    aes(x = x, xend = xend, y = y, yend = yend), inherit.aes = FALSE, colour = "white", linewidth = .25)
  b <- data.frame(arm = rep(keys, 2), type = rep(c("Stay-still", "Prediction"), each = 3),
     value = c(num(r$arms[keys], "energy_distance_standstill"), num(r$arms[keys], "energy_distance_predicted")))
  b$x <- rep(1:3, 2) + rep(c(-.17, .17), each = 3)
  b$color <- c(rep("#C5CCD2", 3), GRAY, ORANGE, GREEN)
  pb <- ggplot(b, aes(x, value, fill = color)) + geom_col(width = .29) + scale_fill_identity() +
    scale_x_continuous(breaks = 1:3, labels = keys) + labs(x = NULL, y = "Energy distance")
  pb <- pb + geom_segment(data = hatch(2.17, .145, b$value[5], .055),
    aes(x = x, xend = xend, y = y, yend = yend), inherit.aes = FALSE, colour = "white", linewidth = .25)
  draw_figure("figure6_mouse_arms", list(panel(pa, "POD7 audit", a),
    panel(pb, "Baseline vs prediction", b)), height = 66)
}

names_all <- c("figure1_workflow", "figure2_wound7_transfer", "figure3_time_axes", "figure4_methods_benchmark", "figure5_donor_curve_stability", "figure6_mouse_arms")
selected <- if (length(requested)) requested else names_all
if (!all(selected %in% names_all)) stop("Unknown figure name")
for (name in selected) get(name, mode = "function")()
manifest <- list(renderer = "R + ggplot2/grid + tikzDevice + XeLaTeX", font = "Arial",
  text_colour = "#000000", text_weight = "bold",
  base_font_pt = 8, tick_font_pt = 7.5, label_font_pt = 11, journal_width_mm = 180,
  figures = figures, sources = sources, R = R.version.string,
  packages = lapply(c("ggplot2", "tikzDevice", "jsonlite", "digest"), function(p) list(package = p, version = as.character(packageVersion(p)))),
  design_sources = setNames(lapply(c("temporal_design.tikz"),
    function(f) digest::digest(file = file.path(ROOT, "manuscripts/r_tikz", f), algo = "sha256")),
    paste0("manuscripts/r_tikz/", c("temporal_design.tikz"))),
  script_sha256 = digest::digest(file = normalizePath(script), algo = "sha256"))
write_json(manifest, file.path(BUILD, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(capture.output(sessionInfo()), file.path(BUILD, "sessionInfo.txt"))
