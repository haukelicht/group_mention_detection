# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Describe labaled data
#' @author Hauke Licht
#' @date   2023-05-28
#

# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)


# TODO: adjust path
data_path <- file.path("replication", "data")
floats_path <- file.path("replication", "paper")
utils_path <- file.path("replication", "code", "r_utils")

# figure setup
source(file.path(utils_path, "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path = figs_dir)

# tables setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)


label_map <- c(
  "1" = "social group",
  "2" = "political group",
  "3" = "political institution",
  "4" = "collective actor",
  "5" = "implicit group reference",
  "6" = "unsure"
)


parse_jsonlines_file <- function(fp) {
  read_lines(fp) %>%
    map(fromJSON) %>% 
    tibble::enframe(name = NULL) %>% 
    unnest_wider(value) %>% 
    select(id, tokens, labels, metadata) %>% 
    mutate(
      labels = map(labels, "BSCModel")
      , n_tokens = lengths(labels)
    ) %>% 
    unnest_wider(metadata)
}

extract_spans <- function(labs, toks) {
  idxs <- labs!=0
  if (sum(idxs) == 0)
    return(tibble(category = NA_character_))
  sidxs <- cumsum(labs==0)[idxs]
  spans <- split(labs[idxs], sidxs)
  toks <- split(toks[idxs], sidxs)
  tibble(
    category = label_map[map_int(spans, 1)-5]
    , text = map_chr(toks, paste, collapse = " ")
    , n_tokens = lengths(spans)
  )
}

# load UK manifesto data ----

fp <- file.path(data_path, "annotation", "labeled", "uk-manifestos_all_labeled.jsonl")

dat <- parse_jsonlines_file(fp)
dat <- mutate(dat, spans = map2(labels, tokens, extract_spans))

spans <- dat %>% 
  select(id, sentence_id, n_tokens_total = n_tokens, spans) %>% 
  unnest(spans) %>% 
  mutate(category = factor(category, label_map))

tmp <- filter(spans, category == "social group")

# share unique
nrow(tmp)/length(unique(tmp$text))

span_counts <- spans %>% 
  group_by(id, sentence_id) %>% 
  summarise(n_spans = list(map_int(label_map, ~sum(category %in% .)))) %>% 
  ungroup() %>% 
  unnest_longer(n_spans) %>% 
  mutate(
    category = factor(n_spans_id, names(label_map), label_map)
    , n_spans_id = NULL
  )

any_spans_shares <- span_counts %>% 
  group_by(category) %>% 
  summarise(
    any_span = mean(n_spans > 0 )
    , n_span = sum(n_spans)
  ) %>% 
  filter(category != "unsure") %>% 
  left_join(
    spans %>% 
      group_by(category) %>% 
      summarise(n_unique = n_distinct(text))
  )
    
tmp <- spans %>% 
  filter(!is.na(category)) %>% 
  group_by(category) %>% 
  summarise(sums = list(summary(n_tokens))) %>% 
  unnest_wider(sums) %>% 
  mutate(across(-1, as.vector)) %>% 
  select(1, 5, 3, 4, 6, 7)

# NOTE: Table B7
left_join(any_spans_shares, tmp) %>% 
  quick_kable(
    caption = paste(
      "Descriptive statistics of group mentions in labeled sentences in our UK party manifesto corpus.",
      collapse = " "
    ),
    col.names = c("Category", "Share any mention", "$N$", "$N_{\\text{unique}}$", "Mean", paste0(25*1:3, "\\% perc."), "Max."),
    label = "mention_descriptives_uk_mans",
    align = c("l", rep("r", 8))
  ) %>% 
  add_header_above(c(" " = 2, "Mentions" = 2, "$N$ tokens" = 5), escape = FALSE) %>% 
  save_kable(.file.name = "tableB7")

# load DE manifesto data ----

fp <- file.path(data_path, "annotation", "labeled", "de-manifestos_all_labeled.jsonl")
dat <- parse_jsonlines_file(fp)

dat <- mutate(dat, spans = map2(labels, tokens, extract_spans))

spans <- dat %>% 
  select(id, sentence_id, n_tokens_total = n_tokens, spans) %>% 
  unnest(spans) %>% 
  mutate(category = factor(category, label_map))

span_counts <- spans %>% 
  group_by(id, sentence_id) %>% 
  summarise(n_spans = list(map_int(label_map, ~sum(category %in% .)))) %>% 
  ungroup() %>% 
  unnest_longer(n_spans) %>% 
  mutate(
    category = factor(n_spans_id, names(label_map), label_map)
    , n_spans_id = NULL
  )

any_spans_shares <- span_counts %>% 
  group_by(category) %>% 
  summarise(
    any_span = mean(n_spans > 0 )
    , n_span = sum(n_spans)
  ) %>% 
  filter(category %in% label_map[1:4]) %>% 
  left_join(
    spans %>% 
      group_by(category) %>% 
      summarise(n_unique = n_distinct(text))
  )
    
tmp <- spans %>% 
  filter(category %in% label_map[1:4]) %>% 
  group_by(category) %>% 
  summarise(sums = list(summary(n_tokens))) %>% 
  unnest_wider(sums) %>% 
  mutate(across(-1, as.vector)) %>% 
  select(1, 5, 3, 4, 6, 7)

# NOTE: Table B8
left_join(any_spans_shares, tmp) %>% 
  quick_kable(
    caption = paste(
      "Descriptive statistics of group mentions in labeled sentences in our German party manifesto corpus.",
      collapse = " "
    ),
    col.names = c("Category", "Share any mention", "$N$", "$N_{\\text{unique}}$", "Mean", paste0(25*1:3, "\\% perc."), "Max."),
    label = "mention_descriptives_de_mans",
    align = c("l", rep("r", 8))
  ) %>% 
  add_header_above(c(" " = 2, "Mentions" = 2, "$N$ tokens" = 5), escape = FALSE) %>% 
  save_kable(.file.name = "tableb8")

# load UK Commons data ----

fp <- file.path(data_path, "annotation", "labeled", "uk-commons_all_labeled.jsonl")

dat <- parse_jsonlines_file(fp)
dat <- mutate(dat, spans = map2(labels, tokens, extract_spans))

spans <- dat %>% 
  select(id, sentence_id, n_tokens_total = n_tokens, spans) %>% 
  unnest(spans) %>% 
  mutate(category = factor(category, label_map))

tmp <- filter(spans, category == "social group")

# share unique
nrow(tmp)/length(unique(tmp$text))

span_counts <- spans %>% 
  group_by(id, sentence_id) %>% 
  summarise(n_spans = list(map_int(label_map, ~sum(category %in% .)))) %>% 
  ungroup() %>% 
  unnest_longer(n_spans) %>% 
  mutate(
    category = factor(n_spans_id, names(label_map), label_map)
    , n_spans_id = NULL
  )

any_spans_shares <- span_counts %>% 
  group_by(category) %>% 
  summarise(
    any_span = mean(n_spans > 0 )
    , n_span = sum(n_spans)
  ) %>% 
  filter(category != "unsure") %>% 
  left_join(
    spans %>% 
      group_by(category) %>% 
      summarise(n_unique = n_distinct(text))
  )

tmp <- spans %>% 
  filter(!is.na(category)) %>% 
  group_by(category) %>% 
  summarise(sums = list(summary(n_tokens))) %>% 
  unnest_wider(sums) %>% 
  mutate(across(-1, as.vector)) %>% 
  select(1, 5, 3, 4, 6, 7)

# TODO: Table B9
left_join(any_spans_shares, tmp) %>% 
  quick_kable(
    caption = paste(
      "Descriptive statistics of group mentions in labeled sentences in our UK \\emph{House of Commons} corpus.",
      collapse = " "
    ),
    col.names = c("Category", "Share any mention", "$N$", "$N_{\\text{unique}}$", "Mean", paste0(25*1:3, "\\% perc."), "Max."),
    label = "mention_descriptives_uk_commons",
    align = c("l", rep("r", 8))
  ) %>% 
  add_header_above(c(" " = 2, "Mentions" = 2, "$N$ tokens" = 5), escape = FALSE) %>% 
  save_kable(.file.name = "tableB9")

