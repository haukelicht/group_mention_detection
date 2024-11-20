# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze classification performance of Dolinksy-Huber-Horne dictionary
#' @author Hauke Licht
#' @date   2024-08-25
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

library(dplyr)
library(tidyr)
library(purrr) 
library(stringr)  
library(ggplot2)  

# TODO: adjust path
floats_path <- file.path("replication", "paper")
results_path <- file.path("replication", "results", "validation", 'dhh_dictionary') 
utils_path <- file.path("replication", "code", "r_utils")

# figure setup
source(file.path(utils_path, "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path = figs_dir)

# table setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)

parse_test_json <- function(fp) {
  fps <- list.files(fp, pattern = "test_results\\.json", full.names = TRUE)
  lines <- suppressWarnings(map_chr(set_names(fps), readLines))
  lines <- unlist(strsplit(lines, split = "(?<=\\})(?=\\{)", perl = TRUE))
  lines <- str_replace_all(lines, ": NaN", ": null")
  tmp <- map(lines, jsonlite::fromJSON)
  tmp <- map(tmp, discard, is.null)
  res <- map_dfr(tmp, as.data.frame)
  res$line_nr <- 1:nrow(res)
  if (!"run_id" %in% names(res)) {
    res$run_id <- sub("-?test_results\\.json", "", basename(fps))
  } else {
    res$run_id <- sub("-?test_results\\.json", "", basename(res$run_id))
  }
  return(res)
}

test_results_to_df <- function(x) {
  grps <- nrow(count(x, run_id, epoch))
  if (grps == 1) {
    x <- pivot_longer(x, -c(run_id))
  } else {
    x <- x |>
      pivot_longer(-c(run_id, epoch)) |>
      separate(run_id, c("repetition", "step"), sep = "-", remove = FALSE)
  }
  x |>
    tidyr::extract(name, c("what", "this", "metric", "kind"), regex = "^test_([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |>
    filter(!is.na(what)) |>
    select(-run_id) |>
    mutate(
      what = ifelse(kind == "", what, paste0(what, kind)),
      kind = NULL
    )
}

eval_schemes <- c("seqeval", "spanlevel_spanwise", "spanlevel_seqavg", "wordlevel", "doclevel")
eval_scheme_map <- c(
  seqeval = "seqeval",
  spanlevel_spanwise = "cross span avg.",
  spanlevel_seqavg = "within sentence avg.",
  wordlevel = "Word level",
  doclevel = "Sentence level"
)
eval_scheme_map_latex <- eval_scheme_map
eval_scheme_map_latex[1] <- "\\texttt{seqeval}"

seqeval_note <- "\\emph{Note:} \\texttt{seqeval} is the strict metric proposed by \\citet{ramshaw_text_1995} and implemented by \\citet{nakayama_seqeval_2018}."

# results in our data ----

experiment_name <- "eval_uk-manifestos"
fp <- file.path(results_path, experiment_name)

test_res <- parse_test_json(fp)

test_res <- test_res |> 
  as_tibble() |> 
  pivot_longer(-c(run_id)) |>
  tidyr::extract(name, c("what", "this", "metric", "kind"), regex = "^([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |>
  filter(!is.na(what)) |>
  rename(fold = run_id) |>
  mutate(
    what = ifelse(kind == "", what, paste0(what, kind)),
    kind = NULL
  )

test_res_sumstats <- test_res |> 
  filter(this == "SG") |> 
  group_by(what, metric) |> 
  summarise(
    mean = mean(value, na.rm = TRUE)
    , sd = sd(value, na.rm = TRUE)
    , q05 = quantile(value, .05, na.rm = TRUE)
    , q95 = quantile(value, .95, na.rm = TRUE)
  ) |> 
  ungroup()

tab_test_res_metrics <- test_res_sumstats |> 
  transmute(
    across(1:2)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  pivot_wider(names_from = "what") |> 
  select(metric, !!eval_schemes) 
  
# NOTE: Table G1
tab_test_res_metrics |>
  mutate(metric = str_to_title(metric)) |> 
  quick_kable(
    caption = paste(
      "Summary of test set performances in of Dolinsky-Huber-Howe dictionary evaluated in our human-annotated UK manifesto sentences.",
      "Values (in brackets) report the average (90\\% quantile range) of performances across 5 folds",
      "Rows report results in terms of the F1-score, precision, and recall.",
      "Columns distinguish between different evaluation schemes.",
      seqeval_note,
      collapse = " "
    )
    , col.names = c("", eval_scheme_map_latex)
    , align = c("l", rep("c", 5))
    , label = paste0(experiment_name, "_all_metrics")
  ) |>
  add_header_above(c(" " = 1, "Mention level" = 3, " " = 2)) |> 
  save_kable(.file.name = "tableG1")


# results in Thau (2019) data ----

experiment_name <- "eval_thau2019-manifestos"
fp <- file.path(results_path, experiment_name)

test_res <- parse_test_json(fp)

test_res <- test_res |> 
  as_tibble() |> 
  pivot_longer(-c(run_id)) |>
  tidyr::extract(name, c("what", "this", "metric", "kind"), regex = "^([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |>
  filter(!is.na(what)) |>
  rename(fold = run_id) |>
  mutate(
    what = ifelse(kind == "", what, paste0(what, kind)),
    kind = NULL
  ) |> 
  filter(
    # note: discard doclevel eval because it makes no sense in Thau data (it only contains positive example sentences due to how we matched annotations in the tabular file to sentence texts)
    what %in% eval_schemes[1:4]
  )

test_res_sumstats <- test_res |> 
  filter(this == "SG") |> 
  group_by(what, metric) |> 
  summarise(
    mean = mean(value, na.rm = TRUE)
    , sd = sd(value, na.rm = TRUE)
    , q05 = quantile(value, .05, na.rm = TRUE)
    , q95 = quantile(value, .95, na.rm = TRUE)
  ) |> 
  ungroup()

tab_test_res_metrics <- test_res_sumstats |> 
  transmute(
    across(1:2)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  pivot_wider(names_from = "what") |> 
  select(metric, !!eval_schemes[-5]) 


# NOTE: Table G2
tab_test_res_metrics |>
  mutate(metric = str_to_title(metric)) |> 
  quick_kable(
    caption = paste(
      "Summary of test set performances in of Dolinsky-Huber-Howe dictionary evaluated in human-annotated UK manifesto sentences in Thau (2019) data.",
      "Values (in brackets) report the average (90\\% quantile range) of performances across 5 folds",
      "Rows report results in terms of the F1-score, precision, and recall.",
      "Columns distinguish between different evaluation schemes.",
      seqeval_note,
      collapse = " "
    )
    , col.names = c("", eval_scheme_map_latex[-5])
    , align = c("l", rep("c", 4))
    , label = paste0(experiment_name, "_all_metrics")
  ) |>
  add_header_above(c(" " = 1, "Mention level" = 3, " " = 1)) |> 
  save_kable(.file.name = "tableG2")


# dictionary expansion ----

experiment_name <- "dictionary_expansion"
fp <- file.path(results_path, experiment_name)

## estimated reviewing effort ----

effort_estimates <- read_csv(file.path(fp, "dictionary_expansion_words_to_review_estimates.csv" ))

# NOTE: Table G3
effort_estimates |> 
  mutate(in_dictionary = factor(in_dictionary, c(T, F), c("yes", "no"))) |> 
  arrange(in_dictionary) |> 
  rename(`among seed keywords` = in_dictionary) |> 
  quick_kable(
    caption = paste(
      "Number of unqiue keywords in set added through word embedding-based expansion",
      "depending on $k$, the number of nearest neighbours considered per ``seed'' keyword.",
      "Rows indicate the number of words in the nearest neighbor set that are/are not among the ``seed'' keywords.",
      collapse = " "
    )
    , align = rep("r", 5)
    , label = paste0(experiment_name, "_effort_estimates")
  ) |>
  add_header_above(c(" " = 1, "$k$" = 4), escape = FALSE) |>
  save_kable(.file.name = "tableG3")

## performance for naive dictionary expansion approach (i.e. without review) ----

dictionaries_map <- c(
  "expanded_dictionary_k10" = "k=10",
  "expanded_dictionary_k25" = "k=25",
  "expanded_dictionary_k50" = "k=50",
  "expanded_dictionary_k100" = "k=100"
)

test_res <- parse_test_json(fp)

test_res <- test_res |> 
  as_tibble() |> 
  pivot_longer(-c(run_id)) |>
  separate(run_id, c("dictionary", "fold"), sep = "-", remove = FALSE) |> 
  tidyr::extract(name, c("what", "this", "metric", "kind"), regex = "^([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |>
  filter(!is.na(what)) |>
  select(-run_id) |>
  mutate(
    what = ifelse(kind == "", what, paste0(what, kind)),
    kind = NULL,
    dictionary = factor(dictionary, names(dictionaries_map), dictionaries_map)
  ) |> 
  filter(
    # note: discard doclevel eval because it makes no sense in Thau data (it only contains positive example sentences due to how we matched annotations in the tabular file to sentence texts)
    what %in% eval_schemes[1:4]
  )

test_res_sumstats <- test_res |> 
  filter(this == "SG") |> 
  group_by(dictionary, what, metric) |> 
  summarise(
    mean = mean(value, na.rm = TRUE)
    , sd = sd(value, na.rm = TRUE)
    , q05 = quantile(value, .05, na.rm = TRUE)
    , q95 = quantile(value, .95, na.rm = TRUE)
  ) |> 
  ungroup()

tab_test_res_f1s <- test_res_sumstats |> 
  filter(metric == "f1") |> 
  transmute(
    across(1:3)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  pivot_wider(names_from = "what") |> 
  select(dictionary, !!eval_schemes[-5])

# NOTE: Table G4
tab_test_res_f1s |>
  quick_kable(
    caption = paste(
      "Summary of test set performances in annotated UK manifesto sentences in the  Thau (2019) data of automatically expanded versions of the Dolinsky-Huber-Howe dictionary in terms of the F1-score.",
      "Values (in brackets) report the average (90\\% quantile range) of performances across 5 folds",
      "Rows report results for differnt values of $k$ that indicates how many nearest neighbor terms per original keyword were included to expand the dictionary.",
      "Columns distinguish between different evaluation schemes (i.e., different ways to compute the F1 score).",
      seqeval_note,
      collapse = " "
    )
    , col.names = c("Category", eval_scheme_map_latex[-5])
    , align = c("l", rep("c", 4))
    , label = paste0(experiment_name, "_all_f1s_by_k")
  ) |>
  add_header_above(c(" " = 1, "Mention level" = 3, " " = 1)) |> 
  save_kable(.file.name = "tableG4")

# NOTE: Figure G3
p <- test_res_sumstats |> 
  filter(
    what %in% c("wordlevel")
  ) |>
  mutate(
    metric = str_to_title(metric)
    , dictionary = factor(dictionary, levels(test_res_sumstats$dictionary), str_remove(levels(test_res_sumstats$dictionary), "k="))
  ) |>
  ggplot(aes(x = dictionary, y = mean)) + 
    geom_linerange(aes(ymin = q05, ymax = q95)) + 
    geom_point(pch = 21, fill = "white") +
    geom_point(size = 0.2) +
    ylim(0, 1) + 
    facet_grid(
      cols = vars(metric)
    ) + 
    labs(
      x = expression(italic(k)),
      y = "Word-level metric"
    )

(pn <- paste0(experiment_name, "_wordlevel_f1_by_k"))
cap <- paste(
  "Summary of test set performances in annotated UK manifesto sentences in the Thau (2019) data of automatically expanded versions of the Dolinsky-Huber-Howe dictionary in terms of the word-level F1-score, precision, and recall.",
  "Values (vertical bars) report the average (90\\% quantile range) of performances across 5 folds",
  "x-axis labels indicate how many ($k$) nearest neighbor terms per original keyword were included to expand the dictionary.",
  seqeval_note,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, "figureG3", cap = cap, w = 4, h = 3)

