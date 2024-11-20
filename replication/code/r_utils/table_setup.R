require(readr, quietly = TRUE)
require(purrr, quietly = TRUE)
require(stringr, quietly = TRUE)
require(knitr, quietly = TRUE)
require(kableExtra, quietly = TRUE)

options(knitr.table.format = "latex")
options(knitr.kable.NA = '')
options(knitr.kable.digits = 3)

dflt_kable_style <- partial(kable_styling, full_width = FALSE, position = "center", font_size = 10L, latex_options = c("hold_position"))
use_kable <- function(x, ...) {
  kab <- kable(x, ..., format = "latex", digits = 3, booktabs = TRUE, linesep = "", escape = FALSE)
  attr(kab, "data") <- x
  return(kab)
}
quick_kable <- function(...) {
  kab <- use_kable(...)
  data <- attr(kab, "data")
  out <- dflt_kable_style(kab)
  attr(out, "data") <- data
  return(out)
}

save_kable <- function(
    x
    , dir
    , position = "!t"
    , overwrite = FALSE
    , .file.name = sub(".*\\\\label\\{([^{]+)\\}.*", "\\1", attr(x, "kable_meta")$caption, perl = TRUE)
    , .file.name.cleaning = c("^tab:" = "")
    , .file.extension = "tex"
    , .write = TRUE
    , .write.data = TRUE
    , .write.data.digits = getOption("knitr.kable.digits", 3)
) {
  if (!is.null(.file.name.cleaning))
    .file.name <- stringr::str_replace_all(.file.name, .file.name.cleaning)
  
  if (.write.data && !is.null(attr(x, "data"))) {
    d <- attr(x, "data")
    idxs <- map_lgl(d, is.double)
    d[idxs] <- map(d[idxs], ~round(., .write.data.digits))
    readr::write_tsv(d, file.path(dir, paste0(.file.name, ".tsv")))
  }
  
  latex <- as.character(x)
  if (!is.null(position))
    latex <- sub("(?<=\\\\begin\\{table\\})(\\[[^]]+\\])?(?=\n)", sprintf("[%s]", position), latex, perl = TRUE)
  
  fp <- file.path(dir, paste0(.file.name, ".", .file.extension))
  if (!file.exists(fp) || overwrite)
    readr::write_lines(latex, fp)
  
  return(x)
}
