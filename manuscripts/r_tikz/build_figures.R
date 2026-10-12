#!/usr/bin/env Rscript
# Rendering only: completed JSON reports -> R/grid -> editable TikZ.
# All dimensions below are millimetres at the final 180 mm journal width.
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(jsonlite)
  library(tikzDevice)
  library(patchwork)
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
DONORS <- c("PWH26" = BLUE, "PWH27" = ORANGE, "PWH28" = TEAL)
font_commands <- c("\\usepackage{fontspec}", "\\setmainfont{Arial}", "\\setsansfont{Arial}")
options(tikzDefaultEngine = "xetex",
        tikzUnicodeMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricPackages = c(font_commands, "\\usetikzlibrary{calc}"),
        tikzMetricsDictionary = file.path(BUILD, "arial-metrics"))

theme_set(theme_classic(base_size = 8, base_family = "Arial") + theme(
  # Only the separate panel letters are bold; ordinary figure text is regular.
  text = element_text(colour = "black", face = "plain"),
  axis.text = element_text(size = 7.5, colour = "black", face = "plain"),
  axis.title.x = element_text(size = 8, colour = "black", face = "plain", margin = margin(t = 1.4, unit = "mm")),
  axis.title.y = element_text(size = 8, colour = "black", face = "plain", margin = margin(r = 1.3, unit = "mm")),
  axis.ticks = element_line(linewidth = 0.25, colour = GRAY),
  axis.ticks.length = unit(1, "mm"),
  axis.line = element_line(linewidth = 0.25, colour = GRAY),
  panel.grid.major.y = element_line(linewidth = 0.18, colour = "#E2E7EB", inherit.blank = TRUE),
  panel.grid.minor = element_blank(),
  plot.margin = margin(t = 1.8, r = 1.2, b = 1.2, l = 1.2, unit = "mm"),
  plot.background = element_rect(fill = "white", colour = NA),
  legend.position = "bottom",
  legend.text = element_text(size = 7, colour = "black", face = "plain"),
  strip.text = element_text(size = 7.5, colour = "black", face = "plain"),
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
read_panel_csv <- function(path) {
  full <- file.path(ROOT, path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  read.csv(full, stringsAsFactors = FALSE)
}
measurement_reports <- c("patient_mapping_equivalence", "representation_benchmark",
  "representation_inference", "patient_unit_representation_benchmark", "design_power_simulation")
provenance_repair_reports <- c(
  time_parameterisation = "outputs/scientific_revision_20260926/provenance_repair/time_parameterisation/report.json",
  donor_transfer_flow = "outputs/scientific_revision_20260926/provenance_repair/donor_transfer_flow/report.json",
  donor_count_curve = "outputs/scientific_revision_20260926/provenance_repair/donor_count_curve/report.json"
)
repaired <- function(name) {
  if (name %in% names(provenance_repair_reports)) return(read_report(provenance_repair_reports[[name]]))
  prefix <- if (name %in% measurement_reports) "outputs/scientific_revision_20260922/measurement/" else "outputs/analysis/"
  read_report(paste0(prefix, name, "/report.json"))
}
historical <- function(name) {
  if (name == "cohort_metadata/gse165816_patient_unit_remap")
    return(read_report("outputs/scientific_revision_20260922/measurement/patient_unit_remap/report.json"))
  if (name == "representation_benchmark") return(repaired(name))
  read_report(paste0("outputs/", name, "/report.json"))
}
num <- function(rows, key) vapply(rows, function(x) as.numeric(x[[key]]), numeric(1))
chr <- function(rows, key) vapply(rows, function(x) as.character(x[[key]]), character(1))
vector <- function(x) as.numeric(unlist(x, use.names = FALSE))
ordered_factor <- function(x) factor(x, levels = unique(x))
panel <- function(plot, title, data) list(plot = plot, title = title, data = data)
zero_h <- function() geom_hline(yintercept = 0, colour = GRAY, linewidth = .25)
zero_v <- function(x = 0) geom_vline(xintercept = x, colour = GRAY, linewidth = .3, linetype = "dashed")
xgrid <- function() theme(panel.grid.major.y = element_blank(),
                         panel.grid.major.x = element_line(linewidth = .18, colour = "#E2E7EB", inherit.blank = TRUE))
legend_titles <- function() theme(legend.title = element_text(size = 7, colour = "black", face = "plain"))
# Wrap axis titles at a parenthetical or into balanced lines; never split "n = 25".
wrap_axis_title <- function(label, width) {
  if (nchar(label) <= width) return(label)
  protected <- gsub(" = ", "\u00a7=\u00a7", label, fixed = TRUE)
  open <- regexpr(" (", protected, fixed = TRUE)
  if (open > 1 && open - 1 <= width && nchar(protected) - open <= width) {
    lines <- c(substr(protected, 1, open - 1), substr(protected, open + 1, nchar(protected)))
  } else {
    greedy <- strwrap(protected, width = width)
    target <- max(ceiling(nchar(protected) / length(greedy)), 8)
    lines <- strwrap(protected, width = target + 1)
    if (length(lines) > length(greedy) || any(nchar(lines) > width)) lines <- greedy
  }
  gsub("\u00a7", " ", paste(lines, collapse = intToUtf8(10)), fixed = TRUE)
}
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
record_figure <- function(name, width, height, panels, heading_centres, label_lefts, heading_tops, panel_widths = rep(width / max(1, length(panels)), length(panels))) {
  data_files <- character()
  for (i in seq_along(panels)) {
    path <- file.path(DATA, paste0(name, "_", LETTERS[i], ".csv"))
    write.csv(panels[[i]]$data, path, row.names = FALSE, na = "")
    data_files <- c(data_files, sub(paste0(ROOT, "/"), "", path, fixed = TRUE))
  }
  figures[[name]] <<- list(width_mm = width, height_mm = height,
      panels = vapply(panels, `[[`, character(1), "title"),
      headings = vapply(panels, `[[`, character(1), "title"),
      panel_labels = if (length(panels) > 1) LETTERS[seq_along(panels)] else character(),
      label_left_mm = label_lefts, heading_top_mm = heading_tops,
      panel_width_mm = panel_widths, heading_alignment = "panel-centred",
      heading_centres_mm = heading_centres,
      sources = active_sources, data = data_files)
  active_sources <<- character()
  message("TikZ: ", name)
}
draw_figure <- function(name, panels, height = 82, widths = NULL, nrow = 1, gap = 5, layout = NULL) {
  ncol <- length(panels) / nrow
  stopifnot(ncol == as.integer(ncol))
  width <- 180; left <- 3; top <- 2; bottom <- 1.5; row_gap <- 4
  if (is.null(widths)) widths <- rep((width - 2 * left - gap * (ncol - 1)) / ncol, ncol)
  stopifnot(abs(sum(widths) + gap * (ncol - 1) - 174) < .01)
  row_h <- (height - top - bottom - row_gap * (nrow - 1)) / nrow
  if (is.null(layout)) {
    layout <- lapply(seq_along(panels), function(i) {
      col <- (i - 1) %% ncol + 1; row <- (i - 1) %/% ncol + 1
      c(left + sum(head(widths, col - 1)) + gap * (col - 1),
        top + (row - 1) * (row_h + row_gap), widths[col], row_h)
    })
  }
  stopifnot(length(layout) == length(panels))
  tikz(file.path(TEX, paste0(name, ".tex")), width = width / 25.4, height = height / 25.4,
       pointsize = 8, standAlone = TRUE, engine = "xetex", sanitize = TRUE,
       timestamp = FALSE, documentDeclaration = "\\documentclass[10pt]{standalone}",
       packages = c("\\usepackage{tikz,graphicx}", font_commands))
  on.exit(dev.off())
  grid.newpage()
  for (i in seq_along(panels)) {
    rectangle <- layout[[i]]
    x <- rectangle[1]; y <- height - rectangle[2]
    panel_w <- rectangle[3]; panel_h <- rectangle[4]
    # Centre every title on its full panel, independently of the left-hand letter.
    # Single-panel figures have a title only; PDF validation checks both objects.
    labelled <- length(panels) > 1
    if (labelled) grid.text(LETTERS[i], unit(x, "mm"), unit(y, "mm"),
              just = c("left", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 11, fontface = "bold", col = "black"))
    grid.text(panels[[i]]$title,
              unit(x + panel_w / 2, "mm"),
              unit(y - .5, "mm"), just = c("centre", "top"),
              gp = gpar(fontfamily = "Arial", fontsize = 9, fontface = "plain", col = "black"))
    if (inherits(panels[[i]]$plot, "vector_panel")) {
      # Direct PDF inclusion keeps the hand-maintained tissue artwork out of
      # ggplot scale/clip handling. The same asset is delivered independently.
      tikzAnnotate(sprintf("\\node[anchor=north west,inner sep=0pt] at (%.6f,%.6f) {\\includegraphics[width=%.6fmm]{../%s.pdf}};",
                           x * 72.27 / 25.4, (y - 6.3) * 72.27 / 25.4,
                           panel_w, panels[[i]]$plot$asset))
      next
    }
    pushViewport(viewport(x = unit(x, "mm"), y = unit(y - 6.3, "mm"),
                          width = unit(panel_w, "mm"), height = unit(panel_h - 6.3, "mm"),
                          just = c("left", "top"), clip = "off"))
    plot <- panels[[i]]$plot
    if (inherits(plot, "ggplot") && !inherits(plot, "patchwork")) {
      # Keep full-size lettering inside short panel rows: wrap the wording,
      # not the artwork. A long rotated y title otherwise crosses the header.
      # About 1.55 mm per 8 pt Arial character after tick-label space.
      y_width <- if (panel_h < 55) 18 else 24
      x_width <- floor((panel_w - 14) / 1.55)
      for (axis in c("x", "y")) {
        label <- plot$labels[[axis]]
        if (is.character(label) && length(label) == 1 && !grepl(intToUtf8(10), label, fixed = TRUE)) {
          plot$labels[[axis]] <- wrap_axis_title(label, if (axis == "y") y_width else x_width)
        }
      }
    }
    grob <- if (inherits(plot, "patchwork")) patchworkGrob(plot) else ggplotGrob(plot)
    grid.draw(grob)
    popViewport()
  }
  lefts <- vapply(layout, function(r) r[1], numeric(1))
  tops <- vapply(layout, function(r) r[2], numeric(1))
  panel_widths <- vapply(layout, function(r) r[3], numeric(1))
  centres <- lefts + panel_widths / 2
  record_figure(name, width, height, panels,
                centres, lefts, tops, panel_widths)
}

# Figure 1 gives the physical sampling object the complete opening row.
# B--D retain measured expression context; E--F retain wide numerical panels.
physical_object_layout <- function() list(
  c(3, 2, 174, 60.3),
  c(3, 68.3, 54.6666667, 74), c(62.6666667, 68.3, 54.6666667, 74),
  c(122.3333333, 68.3, 54.6666667, 74),
  c(3, 148.3, 84.5, 62.2), c(92.5, 148.3, 84.5, 62.2))
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
                lapply(steps, function(s) panel(NULL, s, data.frame(step = s))), heading_centres,
                seq(0, by = 180 / max(1, length(steps)), length.out = length(steps)),
                rep(2, length(steps)), rep(180 / max(1, length(steps)), length(steps)))
}
methods <- c("topic_simplex_theta0", "module_score", "pca", "nmf")
method_names <- c("Fibroblast\ntopic", "Module", "PCA", "NMF")

figure1_workflow <- function() {
  biological_population_figure()
}

figure2_wound7_transfer <- function() {
  h <- repaired("human_temporal_flow"); t <- repaired("donor_transfer_flow"); g <- repaired("donor_geometry_inference")
  a <- data.frame(label = c("Prediction", "Stay-still"), value = c(h$all_donors$predicted, h$all_donors$standstill))
  pa <- columns(a, "Energy distance", c(BLUE, GRAY), c(0, 1.7))
  b <- data.frame(label = names(t$per_donor), value = 100 * num(t$per_donor, "improvement"))
  b$predicted <- num(t$per_donor, "ed_predicted")
  b$unchanged <- num(t$per_donor, "ed_standstill")
  pb <- ggplot(b, aes(y = label)) +
    geom_segment(aes(x = predicted, xend = unchanged, yend = label), colour = GRAY, linewidth = .5) +
    geom_point(aes(x = unchanged, shape = "Unchanged"), colour = GRAY, size = 2) +
    geom_point(aes(x = predicted, shape = "Flow"), colour = NAVY, size = 2) +
    scale_shape_manual(values = c("Unchanged" = 1, "Flow" = 16)) +
    labs(x = "Energy distance (lower is better)", y = NULL) + xgrid()
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
figure4_time_axes <- function() {
  r <- read_report("outputs/scientific_revision_20260925/paired_clock/run/paired_clock/report.json")
  path <- "outputs/scientific_revision_20260925/paired_clock/run/paired_clock/per_donor_seed.csv"
  sources[[path]] <<- digest::digest(file=file.path(ROOT,path),algo="sha256")
  active_sources <<- unique(c(active_sources,path))
  raw <- read.csv(file.path(ROOT,path),stringsAsFactors=FALSE)
  keys <- c("rank", "days", "sqrt_days", "log_days")
  labels <- c("Rank", "Days", "Sqrt days", "Log days")
  by_seed <- aggregate(cbind(flow_ed,centroid_ed,unchanged_source_ed)~axis+seed,raw,mean)
  d <- rbind(data.frame(axis=by_seed$axis,seed=by_seed$seed,method="Shared flow",ed=by_seed$flow_ed),
    data.frame(axis=by_seed$axis,seed=by_seed$seed,method="Centroid translation",ed=by_seed$centroid_ed),
    data.frame(axis=by_seed$axis,seed=by_seed$seed,method="Unchanged source",ed=by_seed$unchanged_source_ed))
  d$axis <- factor(d$axis,levels=keys,labels=labels)
  avg <- aggregate(ed~axis+method,d,mean)
  p <- ggplot(d,aes(axis,ed,colour=method,group=interaction(method,seed)))+
    geom_line(linewidth=.30,alpha=.55)+
    geom_point(aes(shape=factor(seed)),size=1.8)+
    geom_line(data=avg,aes(group=method),linewidth=.75)+
    scale_colour_manual(values=c("Shared flow"=NAVY,"Centroid translation"=TEAL,"Unchanged source"=GRAY))+
    scale_shape_manual(values=c(16,17,15),name="Seed")+
    labs(x="Prespecified time coordinate",y="Mean donor energy distance",colour=NULL)+
    guides(shape=guide_legend(nrow=1,order=1),colour=guide_legend(nrow=1,order=2))+
    legend_titles()+theme(legend.box="vertical",legend.spacing.y=unit(.4,"mm"),legend.spacing.x=unit(.8,"mm"))
  raw$difference <- raw$flow_ed - raw$centroid_ed
  donor <- aggregate(difference ~ axis + donor, raw, mean)
  donor$axis <- factor(donor$axis, levels = keys, labels = labels)
  pb <- ggplot(donor,aes(difference,axis,colour=donor,shape=donor)) + zero_v() +
    geom_point(data = transform(raw, axis = factor(axis, levels = keys, labels = labels)),
      aes(difference, axis, colour = donor, shape = donor),
      position = position_jitter(width = 0, height = .12, seed = 17), size = 1.1, alpha = .5) +
    geom_point(position=position_dodge(width=.45),size=2) +
    scale_colour_manual(values=DONORS) + scale_shape_manual(values=c(16,17,15)) +
    labs(x="Flow minus centroid energy distance",y=NULL) + xgrid() +
    guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  clock_detail <- data.frame(axis = factor(raw$axis, levels = keys, labels = labels),
    donor = raw$donor, difference = raw$difference, seed = raw$seed, kind = "donor_seed")
  donor <- rbind(transform(donor, seed = NA_integer_, kind = "donor_mean"), clock_detail)
  draw_figure("figure4_time_axes", list(panel(p,"Paired clock comparison",d),
    panel(pb,"Donor-level clock effects",donor)),height=86)
}
figure5_methods_benchmark <- function() {
  r <- repaired("donor_conditioned_benchmark")
  baseline <- read_report("outputs/scientific_revision_20260922/population/historical_benchmark/report.json")
  keys <- c("M0_standstill", "M1_shared_cfm", "M2_mean_displacement", "M3_nearest_donor", "M4_conditioned_cfm", "M5_optimal_transport")
  a <- data.frame(label = c("Stay-still", "Shared\nCFM", "Mean\nshift", "Nearest\ndonor", "Cond.\nCFM", "OT", "Target\nmarginal"),
    value = c(num(r$summary[keys], "mean_ed"), baseline$matched_equal_donor$mean))
  a$label <- factor(gsub(intToUtf8(10), " ", a$label), levels = rev(gsub(intToUtf8(10), " ", a$label)))
  details <- do.call(rbind, lapply(seq_along(keys), function(index) {
    rows <- r$per_donor[[keys[index]]]
    data.frame(donor = names(rows), label = levels(a$label)[match(gsub(intToUtf8(10), " ",
      c("Stay-still", "Shared\nCFM", "Mean\nshift", "Nearest\ndonor", "Cond.\nCFM", "OT")[index]), levels(a$label))],
      value = num(rows, "ed"))
  }))
  details <- rbind(details, data.frame(donor = chr(baseline$donor_summary, "held_donor"),
    label = "Target marginal", value = num(baseline$donor_summary, "matched_equal_donor_mean_ed")))
  pa <- ggplot(a,aes(value,label)) + geom_point(colour=NAVY,size=2.1) +
    geom_point(data = details, aes(colour = donor, shape = donor),
      position = position_jitter(width = 0, height = .12, seed = 17), size = 1.3, alpha = .8) +
    scale_colour_manual(values = DONORS) + scale_shape_manual(values = c(16,17,15)) +
    labs(x="Mean donor energy distance",y=NULL) + xgrid()
  a <- rbind(transform(a, donor = NA_character_, kind = "macro_mean"), transform(details, kind = "donor_error"))
  b <- do.call(rbind, lapply(names(r$in_sample), function(k) data.frame(fold = k,
    field = c("Shared", "Conditioned"), value = c(r$in_sample[[k]]$shared, r$in_sample[[k]]$conditioned))))
  b$field <- factor(b$field, levels = c("Shared", "Conditioned"))
  pb <- ggplot(b, aes(field, value, group = fold)) + geom_line(colour = PURPLE, alpha = .6, linewidth = .5) +
    geom_point(colour = PURPLE, size = 1.5) + labs(x = NULL, y = "Energy distance")
  c <- do.call(rbind,lapply(baseline$donor_summary,as.data.frame))
  c$mean <- c$shared_cfm_ed-c$matched_equal_donor_mean_ed
  c$low <- c$shared_cfm_ed-c$matched_equal_donor_max_ed
  c$high <- c$shared_cfm_ed-c$matched_equal_donor_min_ed
  pc <- ggplot(c,aes(mean,held_donor,colour=held_donor,shape=held_donor)) + zero_v() +
    geom_errorbar(aes(xmin=low,xmax=high),orientation="y",width=.15,linewidth=.55) +
    geom_point(size=2) + scale_colour_manual(values=DONORS) + scale_shape_manual(values=c(16,17,15)) +
    labs(x="Flow minus target-marginal ED",y=NULL) + xgrid() + theme(legend.position="none")
  marginal_draws <- read_panel_csv("outputs/scientific_revision_20260922/population/historical_benchmark/draw_scores.csv")
  pc <- pc + geom_point(data = marginal_draws,
    aes(x = shared_minus_equal_donor, y = held_donor, colour = held_donor, shape = held_donor),
    inherit.aes = FALSE, position = position_jitter(width = 0, height = .13, seed = 17), size = .8, alpha = .35)
  c <- merge(c, marginal_draws[, c("held_donor", "draw", "shared_minus_equal_donor")], by = "held_donor")
  d <- do.call(rbind,lapply(names(DONORS),function(donor) data.frame(donor=donor,
    field=c("Shared","Conditioned"),value=c(r$per_donor$M1_shared_cfm[[donor]]$ed,r$per_donor$M4_conditioned_cfm[[donor]]$ed))))
  d$field <- factor(d$field,levels=c("Shared","Conditioned"))
  pd <- ggplot(d,aes(field,value,group=donor,colour=donor,shape=donor)) +
    geom_line(linewidth=.5)+geom_point(size=2)+scale_colour_manual(values=DONORS)+
    scale_shape_manual(values=c(16,17,15))+labs(x="Held-donor errors, not training fit",y="Energy distance")+
    guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  draw_figure("figure5_methods_benchmark", list(panel(pa, "Mean held-donor errors", a),
    panel(pb, "Training donors", b),panel(pc,"Target-marginal comparison",c),
    panel(pd,"Unseen-donor conditioning",d)), height = 130, nrow=2)
}
figure6_donor_curve_stability <- function() {
  # Figure 6 is the common-reference 27-fit audit, not the older subset-specific
  # single-seed donor-count diagnostic. The latter is retained as a provenance
  # repair record and is not substituted here because its estimand differs.
  r <- read_report("outputs/scientific_revision_20260922/population/report.json")
  s <- Filter(function(x) x$method == "Shared CFM", r$curve$by_seed)
  summary <- Filter(function(x) x$method == "Shared CFM", r$curve$summary)[[1]]
  a <- data.frame(k = rep(1:2, each = length(s)), seed = rep(chr(s, "seed"), 2),
    value = c(num(s, "k1_ed"), num(s, "k2_ed")), method = "Shared CFM")
  comparator <- Filter(function(x) x$method %in% c("Mean displacement", "Training-target marginal (exact)"), r$curve$by_seed)
  added <- rbind(data.frame(k = 1, seed = chr(comparator, "seed"), value = num(comparator, "k1_ed"), method = chr(comparator, "method")),
    data.frame(k = 2, seed = chr(comparator, "seed"), value = num(comparator, "k2_ed"), method = chr(comparator, "method")))
  a <- rbind(a, added)
  a$method <- factor(a$method, levels = c("Shared CFM", "Mean displacement", "Training-target marginal (exact)"),
    labels = c("Flow", "Mean shift", "Target marginal"))
  pa <- ggplot(a, aes(k, value, colour = method)) + geom_line(aes(group=interaction(method,seed)), linewidth = .45, alpha = .7) +
    geom_point(aes(shape = seed), size = 1.8) +
    scale_colour_manual(values = c(Flow = NAVY, "Mean shift" = ORANGE, "Target marginal" = TEAL), name = NULL) +
    scale_shape_manual(values = c(16, 17, 15), name = "Seed") +
    scale_x_continuous(breaks = 1:2, labels = c("k = 1", "k = 2"), limits = c(.7, 2.3)) +
    labs(x = NULL, y = "ED in common reference coordinates") +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) +
    theme(legend.title = element_text(size = 7, colour = "black", face = "plain"),
      legend.box = "vertical", legend.box.just = "center", legend.spacing.y = unit(.4, "mm"))
  b <- data.frame(seed = chr(s, "seed"), reduction = num(s, "reduction"),
    low = summary$descriptive_block_range_lower, high = summary$descriptive_block_range_upper,
    mean = summary$mean_reduction)
  pb <- ggplot(b, aes(seed, reduction)) +
    annotate("rect", xmin = -Inf, xmax = Inf, ymin = b$low[1], ymax = b$high[1], fill = "#DFEAF1", colour = NA) +
    geom_hline(yintercept = 0, colour = GRAY, linewidth = .3) +
    geom_hline(yintercept = b$mean[1], colour = NAVY, linewidth = .55) +
    geom_point(size = 2.6, shape = 18, colour = NAVY) +
    labs(x = "Training seed", y = "ED reduction (k=1 − k=2)")
  seed_donor <- read_panel_csv("outputs/scientific_revision_20260922/population/paired_donor_seed_scores.csv")
  seed_donor <- seed_donor[seed_donor$method == "Shared CFM", ]
  seed_donor$seed <- as.character(seed_donor$seed)
  pb <- pb + geom_line(data = seed_donor, aes(group = held_donor, colour = held_donor), linewidth = .35, alpha = .65) +
    geom_point(data = seed_donor, aes(colour = held_donor, shape = held_donor), size = 1.4) +
    scale_colour_manual(values = DONORS) + scale_shape_manual(values = c(16,17,15)) +
    guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
  b <- merge(b, seed_donor[, c("seed", "held_donor", "reduction")], by = "seed", suffixes = c("", "_donor"))
  blocks <- Filter(function(x) x$method=="Shared CFM",r$curve$donor_blocks)
  c <- rbind(data.frame(donor=chr(blocks,"held_donor"),k=1,value=num(blocks,"k1_ed")),
    data.frame(donor=chr(blocks,"held_donor"),k=2,value=num(blocks,"k2_ed")))
  pc <- ggplot(c,aes(k,value,group=donor,colour=donor,shape=donor))+geom_line(linewidth=.5)+
    geom_point(size=2)+scale_colour_manual(values=DONORS)+scale_shape_manual(values=c(16,17,15))+
    scale_x_continuous(breaks=1:2,labels=c("k = 1","k = 2"))+labs(x="Three original donor blocks",y="Energy distance")+
    guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  details <- rbind(data.frame(donor = seed_donor$held_donor, seed = seed_donor$seed, k = 1, value = seed_donor$k1_ed),
    data.frame(donor = seed_donor$held_donor, seed = seed_donor$seed, k = 2, value = seed_donor$k2_ed))
  pc <- pc + geom_line(data = details, aes(group = interaction(donor, seed)), linewidth = .25, alpha = .35) +
    geom_point(data = details, size = 1, alpha = .5)
  c <- rbind(transform(c, seed = NA_character_, kind = "donor_mean"), transform(details, kind = "donor_seed"))
  mixture <- read_report("outputs/scientific_revision_20260923/population_mixture/report.json")
  donors <- mixture$source_comparison$per_donor
  d <- do.call(rbind,lapply(c("real_ed","wrong_a_ed","wrong_b_ed","mixed_wrong_ed"),function(key)
    data.frame(donor=chr(donors,"held_donor"),source=key,value=num(donors,key))))
  d$source <- factor(d$source,levels=c("real_ed","wrong_a_ed","wrong_b_ed","mixed_wrong_ed"),
    labels=c("Real","Other A","Other B","Mixture"))
  pd <- ggplot(d,aes(source,value,group=donor,colour=donor,shape=donor))+geom_line(linewidth=.45)+
    geom_point(size=1.8)+scale_colour_manual(values=DONORS)+scale_shape_manual(values=c(16,17,15))+
    labs(x="Same fitted fields; 200 predicted particles",y="Energy distance")+
    guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  source_fit <- read_panel_csv("outputs/scientific_revision_20260923/population_mixture/source_per_fit.csv")
  fit_details <- do.call(rbind, lapply(c("real_ed", "wrong_a_ed", "wrong_b_ed", "mixed_wrong_ed"), function(key)
    data.frame(donor = source_fit$held_donor, seed = source_fit$seed, source = key, value = source_fit[[key]])))
  fit_details$source <- factor(fit_details$source, levels = c("real_ed", "wrong_a_ed", "wrong_b_ed", "mixed_wrong_ed"),
    labels = c("Real", "Other A", "Other B", "Mixture"))
  pd <- pd + geom_point(data = fit_details, position = position_jitter(width = .08, height = 0, seed = 17), size = 1, alpha = .55)
  d <- rbind(transform(d, seed = NA_integer_, kind = "donor_mean"), transform(fit_details, kind = "donor_seed"))
  observed <- read_report("outputs/scientific_revision_20260925/population_observable/final-01/report.json")
  obs <- do.call(rbind,lapply(Filter(function(x) x$target_set=="original_evaluation",observed$macro),as.data.frame))
  method_keys <- c("persistence_decoded","shared_cfm_decoded","mean_displacement_decoded","training_target_marginal_decoded_exact","persistence_observed","training_target_marginal_observed_exact")
  method_labels <- c("Decoded persistence","Decoded flow","Decoded mean shift","Decoded marginal","Observed persistence","Observed marginal")
  obs$method <- factor(obs$method,levels=rev(method_keys),labels=rev(method_labels))
  per_fit <- read_panel_csv("outputs/scientific_revision_20260925/population_observable/final-01/observed_expression_per_fit.csv")
  per_fit <- per_fit[per_fit$target_set == "original_evaluation" & per_fit$method %in% method_keys, ]
  per_fit$method <- factor(per_fit$method, levels = rev(method_keys), labels = rev(method_labels))
  donor_expression <- aggregate(cbind(total_variation, js_divergence_nats) ~ held_donor + k + method, per_fit, mean)
  observation_plot <- function(metric,label) {
    ggplot(obs,aes(.data[[metric]],method,colour=factor(k),shape=factor(k)))+
      geom_point(data = donor_expression,
        position = position_jitter(width = 0, height = .10, seed = 17), size = .95, alpha = .45) +
      geom_point(position=position_dodge(width=.4),size=1.9)+
      scale_colour_manual(values=c(NAVY,ORANGE),name="Training donors")+
      scale_shape_manual(values=c(16,17),name="Training donors")+
      labs(x=label,y=NULL)+xgrid()+legend_titles()+
      guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  }
  expression_data <- merge(obs, donor_expression, by = c("method", "k"), suffixes = c("", "_donor"))
  draw_figure("figure6_donor_curve_stability", list(panel(pa, "Donor curve", a),
    panel(pb, "Seed stability", b), panel(pc,"Donor-level reductions",c),
    panel(pd,"Source substitution",d),panel(observation_plot("total_variation","Mean-profile total variation"),"Observed-expression errors",expression_data),
    panel(observation_plot("js_divergence_nats","Jensen-Shannon divergence (nats)"),"Second expression metric",expression_data)),height=190,nrow=3)
}
figure7_mouse_arms <- function() {
  r <- repaired("mouse_all_arms_pod7_audit"); keys <- c("NDB", "PDB", "GDB")
  a <- data.frame(label = keys, value = 100 * num(r$arms[keys], "improvement_fraction"))
  pa <- columns(a, "Improvement (%)", c(GRAY, ORANGE, GREEN), c(-6, 58)) + zero_h()
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
  b$arm <- factor(b$arm,levels=keys)
  b$type <- factor(b$type,levels=c("Stay-still","Prediction"))
  pb <- ggplot(b,aes(value,arm,shape=type,group=arm))+
    geom_line(colour=GRAY,linewidth=.5)+geom_point(aes(colour=type),size=2)+
    scale_shape_manual(values=c(1,16))+scale_colour_manual(values=c(GRAY,NAVY))+
    labs(x="Energy distance (lower is better)",y=NULL)+xgrid()+
    guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  counts <- do.call(rbind, lapply(keys, function(key)
    data.frame(arm = key, day = names(r$arms[[key]]$n_cells_by_pod), cells = vector(r$arms[[key]]$n_cells_by_pod))))
  counts$day <- factor(counts$day, levels = c("POD0", "POD2", "POD7", "POD30"))
  counts$arm <- factor(counts$arm, levels = rev(keys))
  pc <- ggplot(counts, aes(day, arm, fill = log10(cells))) + geom_tile(colour = "white", linewidth = .5) +
    geom_text(aes(label = format(cells, big.mark = ",")), size = 7.5 / .pt, family = "Arial", colour = "black") +
    scale_fill_gradient(low = "#F2F5F6", high = "#A9C3CD", guide = "none") +
    labs(x = "One animal per arm and time", y = NULL) +
    theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank())
  diagnostics <- do.call(rbind, lapply(keys, function(key)
    data.frame(arm = key, test = c("Baseline", "Noise", "Ordering"),
      passed = c(r$arms[[key]]$beats_baseline, r$arms[[key]]$passes_noise_gate, r$arms[[key]]$holdout_on_direct_path))))
  diagnostics$arm <- factor(diagnostics$arm, levels = rev(keys))
  diagnostics$test <- factor(diagnostics$test, levels = c("Baseline", "Noise", "Ordering"))
  pd <- ggplot(diagnostics, aes(test, arm, fill = passed)) + geom_tile(colour = "white", linewidth = .5) +
    geom_text(aes(label = ifelse(passed, "Pass", "Fail")), size = 7.5 / .pt, family = "Arial", colour = "black") +
    scale_fill_manual(values = c("TRUE" = "#D1E5E4", "FALSE" = "#ECCFCF"), guide = "none") +
    labs(x = "Computational checks, not validation", y = NULL) +
    theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank())
  draw_figure("figure7_mouse_arms", list(panel(pa, "POD7 audit", a),
    panel(pb, "Baseline vs prediction", b), panel(pc, "Arm-time sampling", counts),
    panel(pd, "Retained diagnostic failures", diagnostics)), height = 111, nrow = 2)
}

source(file.path(ROOT, "manuscripts/r_tikz/biological_figures.R"))
source(file.path(ROOT, "manuscripts/r_tikz/quality_figures.R"))
source(file.path(ROOT, "manuscripts/r_tikz/followup_figures.R"))

names_all <- c("figure1_workflow", "figure2_wound7_transfer", "figure3_interpolation_controls", "figure4_time_axes", "figure5_methods_benchmark", "figure6_donor_curve_stability", "figure7_mouse_arms")
selected <- if (length(requested)) requested else names_all
if (!all(selected %in% names_all)) stop("Unknown figure name")
for (name in selected) get(name, mode = "function")()
manifest <- list(renderer = "R + ggplot2/grid + tikzDevice + XeLaTeX", font = "Arial",
  text_colour = "#000000", text_weight = "regular", panel_label_weight = "bold",
  panel_title_alignment = "panel-centred",
  base_font_pt = 8, tick_font_pt = 7.5, label_font_pt = 11, journal_width_mm = 180,
  figures = figures, sources = sources, R = R.version.string,
  packages = lapply(c("ggplot2", "tikzDevice", "jsonlite", "digest"), function(p) list(package = p, version = as.character(packageVersion(p)))),
  design_sources = setNames(lapply(c("temporal_design.tikz"),
    function(f) digest::digest(file = file.path(ROOT, "manuscripts/r_tikz", f), algo = "sha256")),
    paste0("manuscripts/r_tikz/", c("temporal_design.tikz"))),
  script_sha256 = digest::digest(file = normalizePath(script), algo = "sha256"))
auxiliary <- c("manuscripts/r_tikz/biological_figures.R", "manuscripts/r_tikz/quality_figures.R", "manuscripts/r_tikz/followup_figures.R", "manuscripts/r_tikz/panel_a.tex")
manifest$auxiliary_sources <- setNames(lapply(auxiliary,function(f) digest::digest(file=file.path(ROOT,f),algo="sha256")),auxiliary)
write_json(manifest, file.path(BUILD, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(capture.output(sessionInfo()), file.path(BUILD, "sessionInfo.txt"))
