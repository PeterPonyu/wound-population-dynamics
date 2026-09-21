# Publication panels from saved biological-expansion outputs. No inference here.
bio_csv <- function(study, name) {
  path <- paste0("outputs/biological_expansion/", study, "/", name, ".csv")
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
bio_text <- function(...) annotate("text", ..., family="Arial",fontface="bold",colour="black",size=2.65)

skin_material <- function(description) {
  # Schematic anatomy, deliberately separated from measured data panels.
  dermis <- data.frame(x=c(0,10,10,0),y=c(1.8,1.8,6.9,6.9))
  epidermis <- data.frame(x=c(0,3.8,4.3,3.4,0,6.6,5.7,6.2,10,10),
                         y=c(6.9,6.9,6.1,7.9,7.9,7.9,6.1,6.9,6.9,7.9),piece=rep(1:2,each=5))
  fibro <- do.call(rbind,lapply(seq_len(7),function(i) {
    x=c(1.3,3.1,5.2,7.6,8.7,2.1,6.8)[i];y=c(5.9,4.7,5.0,5.8,3.2,2.7,2.6)[i]
    data.frame(x=x+c(-.55,0,.55,0),y=y+c(-.10,.16,.10,-.16),id=i)
  }))
  immune <- data.frame(x=c(4.6,5.4,4.9,6.0),y=c(6.0,6.3,5.5,5.7))
  ggplot()+geom_polygon(data=dermis,aes(x,y),fill="#F6E9DC",colour="#BDB2A7",linewidth=.3)+
    geom_polygon(data=epidermis,aes(x,y,group=piece),fill="#E7C49B",colour="#9B8068",linewidth=.3)+
    geom_polygon(data=fibro,aes(x,y,group=id),fill=TEAL)+
    geom_segment(aes(x=.1,xend=9.9,y=3.9,yend=3.9),colour="#DBAAA1",linewidth=2.7)+
    geom_segment(aes(x=.1,xend=9.9,y=3.9,yend=3.9),colour="white",linewidth=.8)+
    geom_point(data=immune,aes(x,y),colour=PURPLE,size=1.7)+
    bio_text(x=1.65,y=8.6,label="Epidermis")+
    bio_text(x=6.9,y=8.65,label="Wound bed (schematic)")+
    bio_text(x=8.6,y=4.65,label="Dermis")+
    bio_text(x=1.8,y=5.2,label="Fibroblasts")+
    bio_text(x=6.4,y=7.2,label="Immune cells")+
    bio_text(x=7.8,y=3.35,label="Vessel")+
    bio_text(x=5,y=.8,label=description)+
    coord_cartesian(xlim=c(-.2,10.2),ylim=c(-.1,9.25),clip="off")+
    theme_void(base_family="Arial")+theme(plot.margin=margin(0,1,0,1,"mm"))
}

biological_population_figure <- function() {
  d <- read_report("outputs/biological_expansion/population/report.json")
  a <- bio_csv("population","donor_time_counts")
  pa <- skin_material("3 healthy volunteers / 12 specimens\nSkin, day 1, day 7 and day 30")
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
    theme(panel.grid=element_blank(),strip.background=element_blank(),strip.text=element_text(size=7,face="bold",colour="black"),legend.title=element_text(size=7,face="bold",colour="black"))
  a$cond <- factor(a$cond,levels=names(TIME_LABEL),labels=c("0","1","7","30"))
  a$donor <- factor(a$donor,labels=c("Donor A","Donor B","Donor C"))
  pe <- ggplot(a,aes(cond,donor,fill=cells))+geom_tile(colour="white",linewidth=.8)+
    geom_text(aes(label=cells),family="Arial",fontface="bold",colour="black",size=2.8)+
    scale_fill_gradient(low="#F3F5F6",high="#B4C8D5")+labs(x="Observed day",y=NULL)+
    theme(legend.position="none",panel.grid=element_blank(),axis.line=element_blank(),axis.ticks=element_blank())
  pred <- bio_csv("population","projected_prediction"); paths <- bio_csv("population","projected_model_paths")
  obs <- b[b$cond %in% c("Wound1","Wound7"),]
  pf <- ggplot(obs,aes(x,y,colour=cond))+geom_point(size=.23,alpha=.35)+
    geom_path(data=paths,aes(x,y,group=path),inherit.aes=FALSE,colour=TEAL,linewidth=.35,arrow=arrow(length=unit(.9,"mm"),type="closed"))+
    scale_colour_manual(values=TIME_COL,labels=TIME_LABEL)+labs(x=xlab,y=ylab)+
    guides(colour=guide_legend(override.aes=list(size=1.5,alpha=1)))
  draw_figure("figure1_workflow",list(panel(pa,"Acute skin-wound specimens",a),
    panel(pb,"Observed fibroblast states",b),panel(pc,"Donor structure",b),
    panel(pd,"Measured temporal expression",genes),panel(pe,"Cells in each biopsy",a),
    panel(pf,"Projected model paths",paths)),height=204,nrow=3)
}

figure3_interpolation_controls <- function() {
  s <- bio_csv("population","baseline_summary")
  s$method <- factor(s$method,levels=c("Unchanged source","Independent bridge","Shared flow","Centroid translation"))
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
    scale_colour_manual(values=c(BLUE,ORANGE,TEAL))+labs(x="Withheld time",y="Flow ED minus translation ED")+
    guides(colour=guide_legend(nrow=1))
  panels[[3]] <- panel(pc,"Donor-specific comparison",joined)
  d <- bio_csv("population","baseline_donor_macro");d<-d[d$method=="Shared flow",]
  d$day <- factor(d$holdout,levels=c("Wound7","Wound1"),labels=c("Day 7","Day 1"))
  pd <- ggplot(d,aes(factor(seed),100*improvement,colour=day,group=day))+zero_h()+geom_line(linewidth=.5)+geom_point(size=2)+
    scale_colour_manual(values=c(TEAL,ORANGE))+labs(x="Training seed",y="Improvement over unchanged (%)")
  panels[[4]] <- panel(pd,"Training-seed sensitivity",d)
  draw_figure("figure3_interpolation_controls",panels,height=138,nrow=2)
}
