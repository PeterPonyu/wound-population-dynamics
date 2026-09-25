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

biological_population_figure <- function() {
  d <- read_report("outputs/biological_expansion/population/report.json")
  a <- acute_schedule()
  a$day_label <- factor(a$day, levels=c(0,1,7,30), labels=c("0","1","7","30"))
  pa <- structure(list(asset="panel_a"),class="vector_panel")
  b <- bio_csv("population","atlas_display")
  xlab <- sprintf("PC1 (%.1f%%)",100*d$projection$explained_variance_ratio[[1]])
  ylab <- sprintf("PC2 (%.1f%%)",100*d$projection$explained_variance_ratio[[2]])
  pb <- ggplot(b,aes(x,y,colour=cond))+geom_point(size=.25,alpha=.6)+
    scale_colour_manual(values=TIME_COL,labels=TIME_LABEL,breaks=names(TIME_LABEL))+
    labs(x=xlab,y=ylab)+guides(colour=guide_legend(nrow=1,override.aes=list(size=1.5,alpha=1)))
  pc <- ggplot(b,aes(x,y,colour=donor))+geom_point(size=.25,alpha=.6)+
    scale_colour_manual(values=c(PWH26=BLUE,PWH27=ORANGE,PWH28=TEAL),labels=c("Donor A","Donor B","Donor C"))+
    labs(x=xlab,y=ylab)+guides(colour=guide_legend(nrow=1,override.aes=list(size=1.5,alpha=1)))
  genes <- bio_csv("population","marker_expression")
  genes$cond <- factor(genes$cond,levels=names(TIME_LABEL),labels=c("0","1","7","30"))
  genes$donor <- factor(genes$donor,labels=c("Donor A","Donor B","Donor C"))
  genes$gene <- factor(genes$gene,levels=rev(unique(genes$gene)))
  genes$scaled <- ave(genes$pseudobulk_log1p_cp10k,genes$gene,FUN=function(x) (x-min(x))/max(diff(range(x)),1e-10))
  pd <- ggplot(genes,aes(cond,gene,fill=scaled))+geom_tile()+facet_grid(~donor)+
    scale_fill_gradient(low="#EBEEF0",high=TEAL,breaks=c(0,1))+labs(x="Observed day",y=NULL)+
    guides(fill=guide_colourbar(display="rectangles",title="Gene-wise scaled expression",barwidth=unit(24,"mm"),barheight=unit(2,"mm")))+
    theme(panel.grid=element_blank(),strip.background=element_blank(),strip.text=element_text(size=7,face="plain",colour="black"),legend.title=element_text(size=7,face="plain",colour="black"))
  counts <- bio_csv("population","donor_time_counts")
  counts$cond <- factor(counts$cond,levels=names(TIME_LABEL),labels=c("0","1","7","30"))
  counts$donor <- factor(counts$donor,labels=c("Donor A","Donor B","Donor C"))
  pe <- ggplot(counts,aes(cond,donor,fill=cells))+geom_tile(colour="white",linewidth=.8)+
    geom_text(aes(label=cells),family="Arial",fontface="plain",colour="black",size=2.8)+
    scale_fill_gradient(low="#F3F5F6",high="#B4C8D5")+labs(x="Observed day",y=NULL)+
    theme(legend.position="none",panel.grid=element_blank(),axis.line=element_blank(),axis.ticks=element_blank())
  pred <- bio_csv("population","projected_prediction"); paths <- bio_csv("population","projected_model_paths")
  obs <- b[b$cond %in% c("Wound1","Wound7"),]
  pf <- ggplot(obs,aes(x,y,colour=cond))+geom_point(size=.23,alpha=.35)+
    geom_path(data=paths,aes(x,y,group=path),inherit.aes=FALSE,colour=TEAL,linewidth=.35,arrow=arrow(length=unit(.9,"mm"),type="closed"))+
    scale_colour_manual(values=TIME_COL,labels=TIME_LABEL)+labs(x=xlab,y=ylab)+
    guides(colour=guide_legend(override.aes=list(size=1.5,alpha=1)))
  draw_figure("figure1_workflow",list(panel(pa,"Biopsies and flow",a),
    panel(pb,"Observed fibroblast coordinates",b),panel(pc,"Donor structure",b),
    panel(pd,"Measured temporal expression",genes),panel(pe,"Cells in each biopsy",counts),
    panel(pf,"Projected model paths",paths)),height=204,nrow=3)
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
  panels <- list()
  for (i in 1:2) {
    key <- c("Wound7","Wound1")[i];d <- s[s$holdout==key,]
    p <- ggplot(d,aes(energy_mean,method))+geom_errorbar(aes(xmin=energy_min,xmax=energy_max),orientation="y",width=.2,linewidth=.5,colour=NAVY)+
      geom_point(size=2,colour=NAVY)+labs(x="Mean donor energy distance",y=NULL)+xgrid()
    panels[[i]] <- panel(p,paste("Globally withheld day",c("7","1")[i]),d)
  }
  c <- aggregate(energy_distance~holdout+donor+method,b,mean)
  flow <- c[c$method=="Shared flow",];simple <- c[c$method=="Centroid translation",]
  joined <- merge(flow,simple,by=c("holdout","donor"),suffixes=c("_flow","_translation"))
  joined$difference <- joined$energy_distance_flow-joined$energy_distance_translation
  joined$day <- factor(joined$holdout,levels=c("Wound7","Wound1"),labels=c("Day 7","Day 1"))
  joined$donor <- factor(joined$donor,labels=c("Donor A","Donor B","Donor C"))
  pc <- ggplot(joined,aes(day,difference,colour=donor,group=donor))+zero_h()+geom_line(linewidth=.4)+geom_point(size=2)+
    scale_colour_manual(values=c(BLUE,ORANGE,TEAL))+labs(x="Withheld time",y="Flow minus translation ED")+
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
  panels[[5]] <- panel(pe,"Distribution shape diagnostic",geometry)
  matched <- bio_csv("population","matched_cell_summary","computational_extension")
  matched$label <- factor(paste(gsub("Wound","Day ",matched$holdout),matched$comparator,sep=": "),
                        levels=rev(paste(gsub("Wound","Day ",matched$holdout),matched$comparator,sep=": ")))
  pf <- ggplot(matched,aes(flow_minus_comparator_mean,label))+zero_v()+
    geom_errorbar(aes(xmin=difference_min,xmax=difference_max),orientation="y",width=.2,colour=NAVY,linewidth=.5)+
    geom_point(colour=NAVY,size=1.8)+labs(x="Flow error minus comparator error",y=NULL)+xgrid()
  panels[[6]] <- panel(pf,"Matched cell-count comparison",matched)
  draw_figure("figure3_interpolation_controls",panels,height=156,nrow=3)
}
