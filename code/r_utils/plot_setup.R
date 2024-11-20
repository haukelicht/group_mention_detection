require(readr, quietly = TRUE)
require(ggplot2, quietly = TRUE)
require(extrafont, quietly = TRUE)
require(patchwork, quietly = TRUE)

# loadfonts(device="pdf")
font_family <- "" # "Arial"
theme_set(
  theme_bw() +
    theme(
      plot.subtitle = element_text(size = unit(12, "pt"), hjust = 0.5, family = font_family, color = "black")
      , axis.title = element_text(size = unit(10, "pt"), family = font_family, color = "black")
      , axis.text = element_text(size = unit(9, "pt"), family = font_family, color = "black")
      , text = element_text(size = unit(9, "pt"), family = font_family, color = "black")
      # Legend
      , legend.position =  "bottom"
      , legend.direction = "horizontal"
      , legend.key.height = unit(.1, "in")
      , legend.key.width = unit(.1, "in")
      , legend.title = element_text(hjust = 1, vjust = .5, size = unit(9, "pt"), family = font_family, color = "black")
      , legend.text = element_text(hjust = .5, vjust = .5, size = unit(9, "pt"), family = font_family, color = "black")
      , legend.margin = margin()
      , legend.box.margin = margin()
      # Panel
      , panel.grid.minor = element_blank()
      , panel.grid.major = element_line(color = "lightgrey", linewidth = 0.1)
      # Panel strips
      , strip.text.x = element_text(size = unit(10, "pt"), family = font_family, color = "black")
      , strip.text.y = element_text(angle = 0, size = unit(10, "pt"), family = font_family, color = "black")
      , strip.background = element_rect(fill = NA, color = NA)
      # , strip.background = element_rect(fill = "lightgrey", color = NA)
    )
)

save_plot <- function(p, fn, fig.path, cap = NULL, w = 5.5, h = 4, dev = c("pdf", "png", "tiff"), ...) {
  for (.dev in dev) {
    ggplot2::ggsave(
      plot = p,
      filename = file.path(fig.path, paste0(fn, ".", .dev)),
      device = .dev, 
      width = w,
      height = h,
      dpi = 600,
      units = "in",
      ...
    )
  }
  if (!is.null(cap))
    readr::write_lines(cap, file.path(fig.path, paste0(fn, "-caption.tex")))
  return(invisible(NULL))
}
