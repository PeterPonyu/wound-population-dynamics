# Publication panels from saved biological-expansion outputs. No inference here.
bio_csv <- function(study, name, extension="biological_expansion") {
  path <- paste0("outputs/", extension, "/", study, "/", name, ".csv")
  full <- file.path(ROOT, path)
  sources[[path]] <<- digest::digest(file = full, algo = "sha256")
  active_sources <<- unique(c(active_sources, path))
  read.csv(full, stringsAsFactors = FALSE)
}
LINEAGE <- c(fibroblast="Fibro.", keratinocyte="Keratin.", myeloid="Myeloid",
  t_cell="T cell", b_plasma="B/plasma", endothelial="Blood EC", lymphatic_ec="Lymph EC",
  pericyte_smc="Mural", mast="Mast", melanocyte="Melano.", sweat_gland="Gland")
LINEAGE_COL <- setNames(c(TEAL, ORANGE, PURPLE, BLUE, "#CE79A0", RED,
                         "#B0A043", "#5E6B81", "#7DA14E", "#6F4E37", "#ADAAA5"), names(LINEAGE))
TIME_LABEL <- c(Skin="Day 0",Wound1="Day 1",Wound7="Day 7",Wound30="Day 30")
TIME_COL <- c(Skin=GRAY,Wound1=BLUE,Wound7=ORANGE,Wound30=TEAL)
small_map <- function(p) p + theme(panel.grid=element_blank(), axis.text=element_blank(),
  axis.ticks=element_blank(), axis.line=element_blank(), legend.key.size=unit(2.5,"mm"))
foot_groups <- function(composition) {
  spec <- unique(composition[, c("gsm", "arm")])
  spec$arm <- factor(c("DFU-healer"="Healer","DFU-nonhealer"="Non-healer","Non-diabetic"="Healthy")[spec$arm], levels=names(ARMS))
  spec <- spec[order(spec$arm, spec$gsm), ]
  spec$slot <- ave(seq_len(nrow(spec)), spec$arm, FUN=seq_along)
  spec$column <- (spec$slot - 1) %% 3
  spec$row <- (spec$slot - 1) %/% 3
  spec
}
acute_schedule <- function() {
  expand.grid(donor=factor(c("Donor C","Donor B","Donor A"), levels=c("Donor C","Donor B","Donor A")),
              day=c(0, 1, 7, 30), stringsAsFactors=FALSE)
}

# Display summaries only; no refitting, outcome testing or new resampling.
coordinate_summaries <- function(data, group) {
  do.call(rbind, lapply(split(data, data[[group]]), function(part) {
    data.frame(group = part[[group]][1], x = median(part$x), y = median(part$y),
      xlow = quantile(part$x, .25), xhigh = quantile(part$x, .75),
      ylow = quantile(part$y, .25), yhigh = quantile(part$y, .75), n_display = nrow(part),
      stringsAsFactors = FALSE, row.names = NULL)
  }))
}
coordinate_view <- function(plot, data, xlab, ylab, xmax = 4.3, inset = c(.71, .64, 1, 1)) {
  full <- plot + labs(x = NULL, y = NULL) +
    scale_x_continuous(breaks = c(0, 10)) + scale_y_continuous(breaks = c(0, 10)) +
    theme(legend.position = "none", panel.grid = element_blank(), plot.margin = margin(1,1,1,1,"mm"),
      axis.text = element_text(size = 7, colour = "black"), axis.ticks.length = unit(.6, "mm"),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.border = element_rect(fill = NA, colour = GRAY, linewidth = .25))
  focused <- plot + coord_cartesian(xlim = c(-4.7, xmax), ylim = c(-3.2, 4.8)) +
    labs(x = xlab, y = ylab)
  focused + inset_element(full, left = inset[1], bottom = inset[2], right = inset[3], top = inset[4], align_to = "panel")
}

biological_population_figure <- function() {
  d <- read_report("outputs/biological_expansion/population/report.json")
  a <- acute_schedule()
  a$day_label <- factor(a$day, levels=c(0,1,7,30), labels=c("0","1","7","30"))
  pa <- structure(list(asset="panel_a"),class="vector_panel")
  b <- bio_csv("population","atlas_display")
  xlab <- sprintf("PC1 (%.1f%%)",100*d$projection$explained_variance_ratio[[1]])
  ylab <- sprintf("PC2 (%.1f%%)",100*d$projection$explained_variance_ratio[[2]])
  time_summary <- coordinate_summaries(b, "cond")
  pb <- ggplot(b,aes(x,y,colour=cond))+geom_point(size=.35,alpha=.55)+
    scale_colour_manual(values=TIME_COL,labels=TIME_LABEL,breaks=names(TIME_LABEL))+
    geom_segment(data = time_summary, aes(x = xlow, xend = xhigh, y = y, yend = y, colour = group),
      inherit.aes = FALSE, linewidth = .8) +
    geom_segment(data = time_summary, aes(x = x, xend = x, y = ylow, yend = yhigh, colour = group),
      inherit.aes = FALSE, linewidth = .8) +
    geom_point(data = time_summary, aes(x = x, y = y, colour = group), inherit.aes = FALSE,
      shape = 23, fill = "white", size = 2) +
    labs(x=xlab,y=ylab)+guides(colour=guide_legend(nrow=1,override.aes=list(size=1.5,alpha=1)))
  pb <- coordinate_view(pb, b, xlab, ylab)
  donor_summary <- coordinate_summaries(b, "donor")
  pc <- ggplot(b,aes(x,y,colour=donor))+geom_point(size=.35,alpha=.5)+
    scale_colour_manual(values=c(PWH26=BLUE,PWH27=ORANGE,PWH28=TEAL),labels=c("Donor A","Donor B","Donor C"))+
    geom_segment(data = donor_summary, aes(x = xlow, xend = xhigh, y = y, yend = y, colour = group),
      inherit.aes = FALSE, linewidth = .8) +
    geom_segment(data = donor_summary, aes(x = x, xend = x, y = ylow, yend = yhigh, colour = group),
      inherit.aes = FALSE, linewidth = .8) +
    geom_point(data = donor_summary, aes(x = x, y = y, colour = group), inherit.aes = FALSE,
      shape = 23, fill = "white", size = 2) +
    labs(x=xlab,y=ylab)+guides(colour=guide_legend(nrow=1,override.aes=list(size=1.5,alpha=1)))
  pc <- coordinate_view(pc, b, xlab, ylab)
  genes <- bio_csv("population","marker_expression")
  genes$cond <- factor(genes$cond,levels=names(TIME_LABEL),labels=c("0","1","7","30"))
  genes$donor <- factor(genes$donor,labels=c("Donor A","Donor B","Donor C"))
  genes$gene <- factor(genes$gene,levels=rev(unique(genes$gene)))
  genes$scaled <- ave(genes$pseudobulk_log1p_cp10k,genes$gene,FUN=function(x) (x-min(x))/max(diff(range(x)),1e-10))
  pd <- ggplot(genes,aes(cond,gene,fill=scaled))+geom_tile()+facet_grid(~donor)+
    scale_fill_gradient(low="#EBEEF0",high=TEAL,limits=c(0,1),breaks=c(0,.5,1),labels=c("0","0.5","1"))+labs(x="Observed day",y=NULL)+
    guides(fill=guide_colourbar(display="rectangles",title="Gene-wise scaled expression",title.position="top",title.hjust=.5,barwidth=unit(30,"mm"),barheight=unit(2,"mm")))+
    theme(panel.grid=element_blank(),strip.background=element_blank(),strip.text=element_text(size=7,face="plain",colour="black"),legend.title=element_text(size=7,face="plain",colour="black"))
  counts <- bio_csv("population","donor_time_counts")
  counts$cond <- factor(counts$cond,levels=names(TIME_LABEL),labels=c("0","1","7","30"))
  counts$donor <- factor(counts$donor,levels=c("PWH28","PWH27","PWH26"),labels=c("Donor C","Donor B","Donor A"))
  pe <- ggplot(counts,aes(cond,donor,fill=cells))+geom_tile(colour="white",linewidth=.8)+
    geom_text(aes(label=format(cells,big.mark=",")),family="Arial",fontface="plain",colour="black",size=7.5/.pt)+
    scale_fill_gradient(low="#F3F5F6",high="#B4C8D5")+labs(x="Observed day",y=NULL)+
    theme(legend.position="none",panel.grid=element_blank(),axis.line=element_blank(),axis.ticks=element_blank())
  pred <- bio_csv("population","projected_prediction"); paths <- bio_csv("population","projected_model_paths")
  obs <- b[b$cond %in% c("Wound1","Wound7"),]
  observed_endpoint <- obs[obs$cond == "Wound7", c("x", "y")]
  observed_endpoint$distribution <- "Observed day 7"
  predicted_endpoint <- pred[, c("x", "y")]; predicted_endpoint$distribution <- "Predicted day 7"
  endpoints <- rbind(observed_endpoint, predicted_endpoint)
  endpoint_summary <- coordinate_summaries(endpoints, "distribution")
  pf <- ggplot(obs,aes(x,y,colour=cond))+geom_point(size=.3,alpha=.35)+
    geom_point(data = pred, aes(x,y,colour="Predicted"), inherit.aes = FALSE, shape = 1, size = .5, alpha = .4) +
    geom_path(data=paths,aes(x,y,group=path),inherit.aes=FALSE,colour=TEAL,linewidth=.35,arrow=arrow(length=unit(.9,"mm"),type="closed"))+
    geom_point(data = endpoint_summary, aes(x = x, y = y), inherit.aes = FALSE, shape = 23,
      fill = "white", colour = "black", size = 2.1) +
    scale_colour_manual(values=c(TIME_COL[c("Wound1","Wound7")],Predicted=TEAL),
      breaks=c("Wound1","Wound7","Predicted"),labels=c("Day 1","Day 7","Predicted day 7 and paths"))+labs(x=xlab,y=ylab)+
    guides(colour=guide_legend(override.aes=list(size=1.5,alpha=1,shape=c(16,16,1))))
  pf <- coordinate_view(pf + scale_x_continuous(breaks = c(-5, -2.5, 0, 2.5)), obs, xlab, ylab,
    xmax = 11.2, inset = c(.645, .55, 1, 1))
  endpoint_cdf <- ggplot(endpoints, aes(x, colour = distribution)) + stat_ecdf(linewidth = .55) +
    scale_colour_manual(values = c("Observed day 7" = ORANGE, "Predicted day 7" = TEAL)) +
    scale_x_continuous(breaks = c(-5, 0, 10)) +
    scale_y_continuous(breaks = c(0, .5, 1), labels = c("0", ".5", "1")) +
    labs(x = "PC1", y = "ECDF") + theme(legend.position = "none", panel.grid = element_blank(),
      panel.border = element_rect(fill = NA, colour = GRAY, linewidth = .25), plot.margin = margin(1,1,1,1,"mm"))
  pf <- pf + inset_element(endpoint_cdf + theme(axis.text = element_text(size = 7, colour = "black"),
      axis.title = element_text(size = 7, colour = "black"), axis.ticks.length = unit(.6, "mm"),
      plot.background = element_rect(fill = "white", colour = NA)),
    left = .645, bottom = .0, right = 1, top = .5, align_to = "panel")
  endpoint_data <- rbind(data.frame(kind = "numerical_path", path = paths$path, step = paths$step,
      donor = NA_character_, cond = NA_character_, x = paths$x, y = paths$y),
    data.frame(kind = "predicted_endpoint", path = NA_integer_, step = NA_integer_,
      donor = pred$donor, cond = "Wound7", x = pred$x, y = pred$y),
    data.frame(kind = "observed_display", path = NA_integer_, step = NA_integer_,
      donor = obs$donor, cond = obs$cond, x = obs$x, y = obs$y))
  draw_figure("figure1_workflow",list(panel(pa,"Acute-wound sampling and the population prediction question",a),
    panel(pb,"Observed times",b),panel(pc,"Donor structure",b),
    panel(pd,"Temporal expression",genes),panel(pe,"Cells in each biopsy",counts),
    panel(pf,"Predicted endpoints and paths",endpoint_data)),height=212,nrow=3,layout=physical_object_layout())
}

figure3_interpolation_controls <- function() {
  s <- bio_csv("population","baseline_summary")
  extension <- read_report("outputs/computational_extension/population/report.json")
  geometry <- bio_csv("population","geometry_summary","computational_extension")
  added <- geometry[geometry$method=="Location-scale",]
  s <- rbind(s,data.frame(holdout=added$holdout,method=added$method,energy_mean=added$energy_distance,
             energy_min=added$energy_distance,energy_max=added$energy_distance,
             improvement_mean=1-added$energy_distance/geometry$energy_distance[match(paste(added$holdout,"Unchanged source"),paste(geometry$holdout,geometry$method))]))
  s$method <- factor(s$method,levels=c("Unchanged source","Independent bridge","Shared flow","Location-scale","Centroid translation"))
  b <- bio_csv("population","baseline_by_donor_seed")
  geometry_detail <- bio_csv("population", "geometry_by_donor_seed", "computational_extension")
  panels <- list()
  for (i in 1:2) {
    key <- c("Wound7","Wound1")[i];d <- s[s$holdout==key,]
    p <- ggplot(d,aes(energy_mean,method))+geom_errorbar(aes(xmin=energy_min,xmax=energy_max),orientation="y",width=.2,linewidth=.5,colour=NAVY)+
      geom_point(size=2,colour=NAVY)+labs(x="Mean donor energy distance",y=NULL)+xgrid()
    detail <- geometry_detail[geometry_detail$holdout == key, ]
    detail$method <- factor(detail$method, levels = levels(s$method))
    p <- p + geom_point(data = detail, aes(x = energy_distance, y = method, colour = donor, shape = donor),
      inherit.aes = FALSE, position = position_jitter(width = 0, height = .12, seed = 17), size = 1, alpha = .7) +
      scale_colour_manual(values = DONORS) + scale_shape_manual(values = c(16,17,15)) +
      guides(colour = guide_legend(nrow = 1), shape = guide_legend(nrow = 1))
    d <- merge(d, detail[, c("holdout", "method", "donor", "seed", "energy_distance")], by = c("holdout", "method"), all.x = TRUE)
    panels[[i]] <- panel(p,paste("Globally withheld day",c("7","1")[i]),d)
  }
  c <- aggregate(energy_distance~holdout+donor+method,b,mean)
  flow <- c[c$method=="Shared flow",];simple <- c[c$method=="Centroid translation",]
  joined <- merge(flow,simple,by=c("holdout","donor"),suffixes=c("_flow","_translation"))
  joined$difference <- joined$energy_distance_flow-joined$energy_distance_translation
  joined$day <- factor(joined$holdout,levels=c("Wound7","Wound1"),labels=c("Day 7","Day 1"))
  joined$donor <- factor(joined$donor,levels=names(DONORS))
  pc <- ggplot(joined,aes(day,difference,colour=donor,group=donor))+zero_h()+geom_line(linewidth=.4)+geom_point(size=2)+
    scale_colour_manual(values=DONORS)+labs(x="Withheld time",y="Flow minus translation ED")+
    guides(colour=guide_legend(nrow=1))
  panels[[3]] <- panel(pc,"Donor-specific comparison",joined)
  d <- bio_csv("population","baseline_donor_macro");d<-d[d$method=="Shared flow",]
  d$day <- factor(d$holdout,levels=c("Wound7","Wound1"),labels=c("Day 7","Day 1"))
  pd <- ggplot(d,aes(factor(seed),100*improvement,colour=day,group=day))+zero_h()+geom_line(linewidth=.5)+geom_point(size=2)+
    scale_colour_manual(values=c(TEAL,ORANGE))+labs(x="Training seed",y="Improvement (%)")
  panels[[4]] <- panel(pd,"Training-seed sensitivity",d)
  geometry$method <- factor(geometry$method,levels=levels(s$method))
  geometry$day <- factor(geometry$holdout,levels=c("Wound7","Wound1"),labels=c("Day 7","Day 1"))
  pe <- ggplot(geometry,aes(centered_energy,method,colour=day,shape=day))+
    geom_point(size=1.8,position=position_dodge(width=.35))+scale_colour_manual(values=c(TEAL,ORANGE))+
    labs(x="Energy distance after centering",y=NULL)+xgrid()+guides(colour=guide_legend(nrow=1),shape=guide_legend(nrow=1))
  geometry_detail$method <- factor(geometry_detail$method, levels = levels(s$method))
  geometry_detail$day <- factor(geometry_detail$holdout, levels = c("Wound7", "Wound1"), labels = c("Day 7", "Day 1"))
  pe <- pe + geom_point(data = geometry_detail,
    position = position_jitter(width = 0, height = .12, seed = 17), size = .9, alpha = .4)
  geometry <- merge(geometry, geometry_detail[, c("holdout", "method", "donor", "seed", "centered_energy")],
    by = c("holdout", "method"), suffixes = c("", "_donor_seed"))
  panels[[5]] <- panel(pe,"Distribution shape diagnostic",geometry)
  matched <- bio_csv("population","matched_cell_summary","computational_extension")
  matched$label <- factor(paste(gsub("Wound","Day ",matched$holdout),matched$comparator,sep=": "),
                        levels=rev(paste(gsub("Wound","Day ",matched$holdout),matched$comparator,sep=": ")))
  pf <- ggplot(matched,aes(flow_minus_comparator_mean,label))+zero_v()+
    geom_errorbar(aes(xmin=difference_min,xmax=difference_max),orientation="y",width=.2,colour=NAVY,linewidth=.5)+
    geom_point(colour=NAVY,size=1.8)+labs(x="Flow error minus comparator error",y=NULL)+xgrid()
  draws <- bio_csv("population", "matched_cell_draws", "computational_extension")
  draw_macro <- aggregate(energy_distance ~ holdout + draw + method, draws, mean)
  draw_flow <- draw_macro[draw_macro$method == "Shared flow", c("holdout", "draw", "energy_distance")]
  names(draw_flow)[3] <- "flow_error"
  contrasts <- merge(draw_macro[draw_macro$method %in% c("Centroid translation", "Location-scale", "Independent bridge"), ],
    draw_flow, by = c("holdout", "draw"))
  contrasts$difference <- contrasts$flow_error - contrasts$energy_distance
  contrasts$label <- factor(paste(gsub("Wound", "Day ", contrasts$holdout), contrasts$method, sep = ": "),
    levels = levels(matched$label))
  pf <- pf + geom_point(data = contrasts, aes(x = difference, y = label), inherit.aes = FALSE,
    position = position_jitter(width = 0, height = .12, seed = 17), colour = ORANGE, size = .7, alpha = .4)
  matched <- merge(matched, contrasts[, c("holdout", "method", "draw", "difference")],
    by.x = c("holdout", "comparator"), by.y = c("holdout", "method"))
  panels[[6]] <- panel(pf,"Matched cell-count comparison",matched)
  draw_figure("figure3_interpolation_controls",panels,height=156,nrow=3)
}
