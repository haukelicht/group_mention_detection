# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze classifier performances
#' @author Hauke Licht
#' @date   2023-04-13
#' @update 2024-08-25, 2023-04-16
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# TODO: 
#  - adjust I/O paths
#  - remove figures/tables not reported

# setup ----

library(dplyr)
library(tidyr)
library(purrr) 
library(stringr)  
library(ggplot2)  
library(jsonlite)  

# paths
# TODO: update paths
floats_path <- file.path("replication", "paper") 
classifiers_path <- file.path("replication", "results", "classifiers") 
experiment_results_path <- file.path("replication", "results", "experiments") 

# figure setup
source(file.path("replication", "code", "r_utils", "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path = figs_dir)

# table setup
source(file.path("replication", "code", "r_utils", "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir=tabs_dir, overwrite = TRUE)

parse_validation_jsonl <- function(fp) {
  out <- tibble(run_id = sub("-?dev_results\\.jsonl", "", basename(fp)))
  lines <- readLines(fp)
  lines <- str_replace_all(lines, ": NaN", ": null")
  tmp <- map(lines[-length(lines)], jsonlite::fromJSON)
  tmp <- map(tmp, discard, is.null)
  tmp <- map_dfr(tmp, as.data.frame)
  return(bind_cols(out, tmp))
}

parse_validation_jsonls <- function(fp) {
  fps <- list.files(fp, pattern = "dev_results\\.jsonl", full.names = TRUE)
  map_dfr(fps, parse_validation_jsonl)
}

validation_results_to_df <- function(x) {
  x |> 
    pivot_longer(-c(run_id, epoch)) |> 
    separate(run_id, c("repetition", "step"), sep = "-", remove = FALSE) |> 
    extract(name, c("what", "this", "metric", "kind"), regex = "^eval_([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |> 
    filter(!is.na(what)) |> 
    group_by(run_id, epoch, what, this, metric, kind) |>
    arrange(run_id, desc(what), this, metric, epoch) |> 
    ungroup() |> 
    mutate(
      what = ifelse(kind == "", what, paste0(what, kind)),
      kind = NULL
    ) |> 
    pivot_wider(names_from = "metric")
}

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

parse_test_jsonl <- function(fp) {
  fps <- list.files(fp, pattern = "test_results\\.json", full.names = TRUE)
  lines <- suppressWarnings(map(set_names(fps), readLines))
  lines <- unlist(lines)
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
    extract(name, c("what", "this", "metric", "kind"), regex = "^test_([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |> 
    filter(!is.na(what)) |> 
    select(-run_id) |> 
    mutate(
      what = ifelse(kind == "", what, paste0(what, kind)),
      kind = NULL
    )
}

transfer_test_results_to_df <- function(x) {
  x |> 
    filter(!grepl("adapt\\d{2}-extra$", run_id)) |> 
    select(-epoch) |> 
    pivot_longer(-c(run_id, line_nr)) |> 
    separate(run_id, c("repetition", "fold"), sep = "-", remove = FALSE, extra = "merge") |> 
    mutate(domain = ifelse(fold == "baseline", "source", "target")) |> 
    extract(name, c("tmp", "what", "this", "metric", "kind"), regex = "^(eval|test)_([a-z]+)\\.([a-zA-Z]+)_([a-z1]+)(_[a-z]+)?$") |> 
    mutate(this = factor(this, types)) |> 
    select(-tmp) |> 
    mutate(
      what = ifelse(kind == "", what, paste0(what, kind)),
      kind = NULL
    ) |> 
    filter(!is.na(what), !is.na(value)) |>
    arrange(desc(what), this, metric, domain, fold, repetition) |> 
    group_by(across(-value)) |> 
    filter(line_nr == max(line_nr)) |> 
    ungroup() |> 
    pivot_wider(names_from = "metric")
}

types <- c("macro", "micro", "SG", "PG", "PI", "ORG", "ISG")

eval_schemes <- c("seqeval", "spanlevel_spanwise", "spanlevel_seqavg", "doclevel")
eval_scheme_map <- c(
  seqeval = "seqeval", 
  spanlevel_spanwise = "cross-span avg.", 
  spanlevel_seqavg = "within-sentence avg.", 
  doclevel = "Sentence level",
  wordlevel = "Word level"
)
eval_scheme_map_latex <- eval_scheme_map
eval_scheme_map_latex[1] <- "\\texttt{seqeval}"

seqeval_note <- "\\emph{Note:} \\texttt{seqeval} is the strict metric proposed by \\citet{ramshaw_text_1995} and implemented by \\citet{nakayama_seqeval_2018}."

# uk-manifestos_model-comparison ----

experiment_name <- "uk-manifestos_model-comparison"
fp <- file.path(experiment_results_path, experiment_name, "results.json")

res <- read_json(fp)

seconds_to_ts <- Vectorize(function(s) {
  hours = s %/% (60*60)
  s = s - hours*60*60
  minutes = s %/% 60
  s = s - minutes*60
  sprintf("%02d:%02d:%02d", as.integer(hours), as.integer(minutes), as.integer(s))
})

search_space <- c(
  "learning rate $\\in \\{9e^{-6}, 2e^{-5}, 4e^{-5}\\}$",
  "training batch size $\\in \\{8, 16, 32\\}$",
  "and weight decay $\\in \\{0.01, 0.1, 0.3\\}$"
)

# NOTE: Table E1
res |> 
  tibble::enframe() |> 
  unnest_wider(value) |> 
  mutate(
    # convert seconds to MM:SS format
    elapsed = seconds_to_ts(elapsed),
    learning_rate = formatC(learning_rate, format = "e", digits = 0),
    learning_rate = paste0("$", sub("e", "e^{", learning_rate), "}$")
  ) |> 
  select(name, 5, 6, 2:4) |> 
  quick_kable(
    caption = paste(
      "Results of model comparions and hyper-parameter search of token classifier fine-tuned for social group mention detection in UK manifestos dataset.",
      "Table reports development set results of best model per pre-fine-tuned model in terms of the \\texttt{seqeval} F1 score.",
      "For each model, we searched the following hyper-parameters for three trials with the TPE (Tree-structured Parzen Estimator) algorithm:", paste(search_space, collapse = ", "),
      seqeval_note,
      collapse = " "
    )
    , col.names = c(" ", "F1", "total time elapsed", "learning rate", "batch size", "weight decay")
    , align = c("l", rep("c", 5))
    , label = experiment_name
  ) |> 
  add_header_above(c(" " = 3, "Best hyper-parameters" = 3)) |> 
  save_kable(.file.name = "tableE1", position='!h')

# uk-manifestos_5x5-crossval_deberta-finetuning ----

experiment_name <- "uk-manifestos_5x5-crossval_deberta-finetuning" 
fp <- file.path(experiment_results_path, experiment_name)

tmp <- read_tsv(file.path(fp, "split_sizes.tsv"))
reframe(tmp, across(-1, range))

test_res <- parse_test_json(fp)

test_res <- test_results_to_df(test_res)

test_res_sumstats <- test_res |> 
  group_by(what, this, metric) |> 
  summarise(
    mean = mean(value, na.rm = TRUE)
    , sd = sd(value, na.rm = TRUE)
    , q05 = quantile(value, .05, na.rm = TRUE)
    , q95 = quantile(value, .95, na.rm = TRUE)
  ) |> 
  ungroup()

tab_test_res_f1s <- test_res_sumstats |> 
  filter(
    !this %in% c("O")
    , metric == "f1"
  ) |> 
  transmute(
    across(1:3)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  mutate(
    across(this, \(x) factor(x, types))
  ) |> 
  pivot_wider(names_from = "what") |> 
  arrange(this) |> 
  select(this, !!eval_schemes)

# NOTE: Table E2
tab_test_res_f1s |> 
  filter(this %in% types[2:7]) |> 
  quick_kable(
    caption = paste(
      "Summary of test set performances in terms of the F1 score of DeBERTa group mention detection classifiers fine-tuned and evaluated on our corpus of labeled UK manifesto sentences.",
      "Values  (in brackets) report the average (90\\% quantile range) of performances of 25 different classifiers fine-tuned in a 5-times repeated 5-fold cross-validation scheme.",
      "Rows report results for the different group categeries included in our coding scheme.",
      "The ``micro'' metric reports results when treating different group types as one.",
      "Columns distinguish between different evaluation schemes (i.e., different ways to compute the F1 score).",
      seqeval_note,
      collapse = " "
    )
    , col.names = c("Category", eval_scheme_map_latex[-5])
    , align = c("l", rep("c", 4)) 
    , label = paste0(experiment_name, "_testset_by_cat")
  ) |> 
  add_header_above(c(" " = 1, "Mention level" = 3, " " = 1)) |> 
  save_kable(.file.name = "tableE2", position="!h")

tab_test_res_sg <- test_res_sumstats |> 
  filter(
    this %in% c("micro", "SG")
    # , metric == "f1"
    , what %in% eval_schemes
  ) |> 
  transmute(
    across(1:3)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  pivot_wider(names_from = "what") |> 
  select(this, metric, !!eval_schemes) 

# Table 2
tab_test_res_sg |> 
  filter(this == "SG") |> 
  select(-this) |>
  arrange(metric) |> 
  mutate(
    metric = str_to_title(metric)
  ) |> 
  quick_kable(
    caption = paste(
      "Summary of test set performances of DeBERTa group mention detection classifiers fine-tuned and evaluated on our corpus of labeled UK manifesto sentences.",
      "Values (in brackets) report the average (90\\% quantile range) of performances of 25 different classifiers fine-tuned in a 5-times repeated 5-fold cross-validation scheme.",
      "Columns distinguish between different evaluation schemes (i.e., different ways to compute the eval. metrics).",
      seqeval_note,
      collapse = " "
    )
    , col.names = c(" ", eval_scheme_map_latex[-5])
    , align = c("l", rep("c", 4))
    , label = paste0(experiment_name, "_testset_sg")
    # TODO: rename table2
  ) |> 
  add_header_above(c(" " = 1, "Mention level" = 3, " " = 1)) |> 
  save_kable(.file.name = "table2", position="!th")

# uk-manifestos_roberta ----

# NOTE: Classifier used for analysis in paper section 4.2 onwards

experiment_name <- "uk-manifestos_roberta"
fp <- file.path(classifiers_path, experiment_name)

test_res <- parse_test_json(fp)
test_res <- test_results_to_df(test_res) 

tab_test_res_f1s <- test_res |> 
  filter(
    grepl("^f1", metric)
    , !this %in% c("O")
  ) |> 
  mutate(
    across(this, \(x) factor(x, types))
  ) |> 
  pivot_wider(names_from = "what") |> 
  arrange(this) |> 
  select(this, !!eval_schemes)

tab_test_res_sg <- test_res |> 
  filter(
    this == "SG"
    , what %in% eval_schemes
  ) |> 
  pivot_wider(names_from = "what") |> 
  select(this, metric, !!eval_schemes)

tab_test_res_sg

## error analysis ----

testset_preds <- read_tsv(file.path(fp, "evaluation_in_testset.tsv"))

# SG recall by span length
tmp <- testset_preds |> 
  filter(type == "SG") |> 
  mutate(
    n_words = stringi::stri_count_words(text, locale = "en_UK")
  )

bins <- c(0, quantile(tmp$n_words, c(.5, .80, .90, 1.0)))

set.seed(1234)
tmp <- tmp |> 
  mutate(
    words_range = cut(n_words, breaks = bins)
  ) |> 
  group_by(words_range) |> 
  summarise(
    n_spans = n()
    # bootstrap salience estimate
    , bs = list(replicate(100, mean(sample(recall, replace = TRUE))))
  ) |> 
  mutate(
    q05 = map_dbl(bs, quantile, 0.05)
    , mean = map_dbl(bs, mean)
    , q95 = map_dbl(bs, quantile, 0.95)
    , bs = NULL
    , words_range = str_replace_all(words_range, c("," = ", "))
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  select(words_range, value, n_spans) 

ptiles <- str_remove(names(bins[2:(length(bins)-1)]), "%")
ptiles_str <- paste(ptiles[-length(ptiles)], collapse = ", ")
ptiles_str <- paste0(ptiles_str, ", and ", ptiles[length(ptiles)])

# NOTE: Table E3
tmp |>
  quick_kable(
    caption = paste(
      "Average mention-level social group mention recall in test set of RoBERTa group mention detection classifier fine-tuned and evaluated on corpus of labeled UK manifesto sentences by number of words in span.",
      sprintf("\\emph{Notes:} Test-set spans grouped into bins based on the number of words they contain at the %s\\%% percentile values.", ptiles_str),
      "For example, bin (0, 2] contains social group mentions that span up to two words, bin (2, 6] mentions that span between three to six words, etc.",
      "Recall values in square brackets report the 90\\% confidence intervall computed from bootstrapped values.",
      # "Estimates for bin (12, 17\\] are only based on 6 spans. ",
      collapse = " "
    )
    , col.names = c("Words in span", "Average mention-level recall", "$N_{\\text{span}}$")
    , label = paste0(experiment_name, "_testset_sg_recall_by_span_length")
    , align = c("c", "r", "r")
  ) |>
  save_kable(.file.name = "tableE3", position = "!h")

## validation against Thau (2017) annotations ----

tmp <- read_tsv(file.path(fp, "evaluation_in_thau2019-manifesto-annotations.tsv"))

set.seed(1234)
thau_sg_recall_sums <- tmp |> 
  group_by(thau_category) |> 
  summarise(
    n_spans = n()
    # bootstrap salience estimate
    , bs = list(replicate(100, mean(sample(recall, replace = TRUE))))
  ) |> 
  mutate(
    q05 = map_dbl(bs, quantile, 0.05)
    , mean = map_dbl(bs, mean)
    , q95 = map_dbl(bs, quantile, 0.95)
  ) 

thau_cats <- c(
  'Age/generation',
  'Economic class',
  'Ethnicity/race',
  'Gender',
  'Geography',
  'Health',
  'Nationality',
  'Religion',
  'Other',
  'none'
)

# NOTE: Figure F1
p <- thau_sg_recall_sums |> 
  mutate(
    thau_category = factor(thau_category, rev(thau_cats))
    , label_ = sprintf("%0.2f (%d)", mean, n_spans)
  ) |> 
  ggplot(aes(y = thau_category, x = mean, xmin = q05, xmax = q95, label = label_)) + 
  geom_linerange(show.legend = FALSE)+
  geom_point(pch = 21, fill = "white", show.legend = FALSE) + 
  geom_point(size = .2) + 
  geom_text(
    nudge_y = .35
    , size = 7/.pt
    # , hjust = 0
    # , vjust = .4
  ) + 
  xlim(0,1) +
  labs(
    y = NULL
    , x = "Average of mention-level recall"
  )

p
pn <- paste0(experiment_name, "_recall-sg_thau-spans")
cap <- paste(
  "Average mention-level recall of social group mentions predicted by RoBERTa token classifier in group-based appeals manually identified by \\citet{thau_how_2019}.",
  "Recall computed by assuming that group mentions identified by \\citet{thau_how_2019} are ``true'' \\emph{social} group mentions",
  "and comparing them to token-level labels predicted by our group mention detection classifier fine-tuned on labeled sentences from all UK party manifestos in our sample.",
  "The x-axis indicates the average share of tokens in a ``true'' mention the classifier has predicted correctly",
  "(values with number of spans plotted above points for readability).",
  "The y-axis indicates the type of group according to Thau's categorization.",
  # "The x-axis reports the number of training samples used to train the classifier.",
  "Horizontal lines report the 90\\% confidence intervall computed from bootstrapped recall values.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figureE1", cap = cap, w = 4.5, h = 3)


# NOTE: Table F3: top-10 terms in spans without any predicted SG tokens
these <- c(
  'Economic class',
  'Geography',
  'Religion',
  'none'
)

tmp |> 
  filter(
    thau_category %in% these
    , recall == 0.0
  ) |> 
  mutate(thau_category = factor(thau_category, these)) |> 
  select(thau_category, text) |> 
  separate_rows(text, sep = "\\s+") |> 
  filter(!text %in% stopwords::stopwords(language = "en")) |> 
  count(thau_category, text) |> 
  arrange(thau_category, desc(n)) |> 
  group_by(thau_category) |> 
  top_n(10, wt = n) |> 
  summarise(toks = paste(sprintf("%s (%d)", text, n), collapse = ", ")) |> 
  quick_kable(
    caption = paste(
      "Most frequent words in group-based appeals manually identified by \\citet{thau_how_2019} our RoBERTa token classifier has not predicted to belong to a social group mention.",
      "One row per type of group according to Thau's categorization, focusing on those where our classifier perfroms relatively poorly in terms of the social group recall.",
      "\\emph{Note:} Values in parentheses indicate the number of occurences in the relevant subset of Thau's data.",
      collapse = " "
    )
    , col.names = c("Thau's categorization", "Word ($N$)") 
    , label = paste0(experiment_name, "_missed_thau-spans_wordfreqs")
  ) |> 
  column_spec(2, width = "4in") |> 
  save_kable(.file.name = "tableF3", position = "!h")


# uk-manifestos_training-size_roberta-finetuning ----

experiment_name <- "uk-manifestos_training-size_roberta-finetuning"
fp <- file.path(experiment_results_path, experiment_name)

test_res <- parse_test_json(fp)
test_res <- test_results_to_df(test_res)

test_res_sumstats <- test_res |> 
  group_by(step, what, this, metric) |> 
  summarise(
    mean = mean(value, na.rm = TRUE)
    , sd = sd(value, na.rm = TRUE)
    , q05 = quantile(value, .05, na.rm = TRUE)
    , q95 = quantile(value, .95, na.rm = TRUE)
  ) |> 
  ungroup()

tab_test_res_sg <- test_res_sumstats |> 
  filter(
    this %in% c("SG", "micro")
    , metric == "f1"
    , what %in% eval_schemes
  ) |> 
  transmute(
    this
    , what
    , training_size = factor(step, paste0("step0", 1:6), 1:6*1000)
    , value = sprintf("%.02f [%.02f, %.02f]", mean, q05, q95)
  ) |> 
  pivot_wider(names_from = "what") |> 
  arrange(this, training_size) |> 
  select(this, training_size, !!eval_schemes[-5])

# NOTE: Figure E1
p <- test_res_sumstats |> 
  filter(
    this %in% c("SG", "micro")
    , metric == "f1"
    # , what %in% eval_schemes
    , what %in% "seqeval"
  ) |> 
  mutate(
    this = ifelse(this == "SG", "Social group", this)
    , what = factor(what, names(eval_scheme_map), eval_scheme_map)
    , step = factor(step, paste0("step0", 1:6), 1:6*1000)
  ) |>
  ggplot(
    aes(
      # y = reorder(what, desc(what)), 
      y = step, 
      x = mean, 
      xmin = q05, 
      xmax = q95, 
      color = this
    )
  ) + 
    geom_linerange(position = position_dodge(.4), show.legend = FALSE)+
    geom_point(position = position_dodge(.4), pch = 21, fill = "white", show.legend = FALSE) + 
    geom_point(position = position_dodge(.4), size = .2) + 
    scale_color_manual(
      breaks = c("Social group", "micro")
      , values = c("black", "#656565") 
    ) +  
    coord_flip() +
    xlim(0.5, 1) + 
    guides(color = guide_legend(override.aes = list(size = 1, pch = 19))) +
    labs(
      x = "F1 (seqeval)"
      , y = "No. training samples"
      , color = NULL
    )

p
pn <- paste0(experiment_name, "_testset_seqeval_f1")
cap <- paste(
  "Summary of test set performances of RoBERTa group mention detection classifier fine-tuned and evaluated on corpus of labeled UK manifesto sentences", 
  "as function of the training data size.",
  "The y-axis indicates the \\texttt{seqeval} F1 score achieved.",
  "The x-axis indicates the number of sentences in the training set.",
  "Points (line ranges) report the average (90\\% quantile range) of performances of 5 different classifiers fine-tuned with different random seeds.",
  "Colors distinguish between the micro performance and the social group mention category-specific performances.",
  seqeval_note,
  # "The micro scores are computed by ingoring differences between group types in our coding scheme.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figureE1", cap = cap, w = 4, h = 3)


# Transfer experiments ----

## Cross-party transfer (witin UK manifestos) ----

experiment_name <- "uk-manifestos_cross-party-transfer_deberta-finetuning"

fp <- file.path(experiment_results_path, experiment_name)

split_sizes <- read_tsv(file.path(fp, "split_sizes.tsv"))

test_res <- parse_test_jsonl(fp)

test_res <- transfer_test_results_to_df(test_res)

tmp <- split_sizes |> 
  filter(grepl("adapt", run_id)) |> 
  select(run_id, n_train) |> 
  separate(run_id, c("repetition", "fold"), sep = "-", remove = FALSE, extra = "merge") |> 
  distinct(fold, n_train) |> 
  mutate(n_train = cumsum(n_train)) |> 
  with(setNames(fold, n_train))

folds_map <- c(
  "test set" = "baseline", 
  "0" = "baseline-extra", 
  tmp
)
folds_map_inv <- set_names(names(folds_map), folds_map)

# NOTE: Figure 4(a)
p <- test_res |>
  filter(
    this %in% "SG" # c("micro", "SG")
    , what %in% "seqeval" # whats[1:2]
  ) |> 
  mutate(
    fold = factor(fold, folds_map, names(folds_map))
  ) |> 
  group_by(what, this, domain, fold) |> 
  summarise(across(f1:recall, list(mean = mean, sd = sd))) |> 
  ggplot(
    aes(
      x = fold
      , y = f1_mean
      , ymin=f1_mean-f1_sd
      , ymax=f1_mean+f1_sd
    )
  )  + 
  geom_linerange(show.legend = FALSE) + 
  geom_point(pch = 21, fill = "white", show.legend = FALSE) + 
  geom_point(size = .2) + 
  ylim(c(0.4, 1)) +
  facet_grid(
    cols = vars(domain)#vars(paste0(domain, "\ndomain"))
    # , rows = vars(what)
    , space = "free"
    , scales =  "free"
  ) + 
  labs(
    y = "F1", 
    x = "              No. training samples"
  ) + 
  theme(
    axis.title.y.left = element_text(angle = 0, vjust = .5)
    # , axis.text.x.bottom =element_text(angle = 35, hjust = 1, vjust = 1)
    , strip.clip = "off" # might requires ggplot2 dev version (see https://stackoverflow.com/a/73106205)
  ) 

p
(pn <- paste0(experiment_name, "_testset"))
cap <- paste(
  "Summary of test set performances of DeBERTa group mention detection classifiers in cross-party transfer experiment.",
  "The y-axis indicates performance in terms of the \\texttt{seqeval} F1 score of classifiers fine-tuned on annotated manifesto sentences from the Labour and Conservative party and then evaluated on, and successively adapted to sentences from other UK parties' manifestos.",
  "Points (line ranges) report the average ($\\pm$\\,1\\,std.\\,dev.) of performances of 5 different classifiers fine-tuned with different random seeds.",
  "Plot panels indicate whether the classifier was tested in the source domain (i.e., Lab/Con manifestos, left)",
  "or the target domain (i.e., DUP, Greens, LibDem, and UKIP manifestos, right).",
  "For the ``target'' panel, the x-axis reports whether the classifier fine-tuned on Lab/Con manifestos was evaluated without adapting it to the target texts (0, i.e., ``zero shot'')",
  "or, if not, how much labeled sentences were used to adapt the classifier (through continued training) befor evaluation.",
  # "Point shapes indicate evaluation scheme (i.e., different ways to compute the F1 score).",
  seqeval_note,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, "figure4a", cap = cap, w = 3, h = 2)


## Cross-lingual transfer (from UK to German manifestos ----

experiment_name <- "manifestos_cross-lingual-transfer_roberta-finetuning"

fp <- file.path(experiment_results_path, experiment_name)

split_sizes <- read_tsv(file.path(fp, "split_sizes.tsv"))

test_res <- parse_test_jsonl(fp)
test_res <- transfer_test_results_to_df(test_res)

tmp <- split_sizes |> 
  filter(grepl("adapt", run_id)) |> 
  select(run_id, n_train) |> 
  separate(run_id, c("repetition", "fold"), sep = "-", remove = FALSE, extra = "merge") |> 
  group_by(fold) |> 
  summarise(n_train = median(n_train)) |> 
  mutate(n_train = cumsum(n_train)) |> 
  with(setNames(fold, n_train))

folds_map <- c(
  "test set" = "baseline", 
  "0" = "baseline-extra", 
  tmp
)
folds_map_inv <- set_names(names(folds_map), folds_map)

# NOTE: Figure 4(b)
p <- test_res |>
  filter(
    this %in% "SG" # c("micro", "SG")
    , what %in% "seqeval" # whats[1:2]
  ) |> 
  mutate(
    fold = factor(fold, folds_map, names(folds_map))
  ) |> 
  group_by(what, this, domain, fold) |> 
  summarise(across(f1:recall, list(mean = mean, sd = sd))) |> 
  ggplot(
    aes(
      x = fold
      , y = f1_mean
      , ymin=f1_mean-f1_sd
      , ymax=f1_mean+f1_sd
    )
  )  + 
  geom_linerange(show.legend = FALSE) + 
  geom_point(pch = 21, fill = "white", show.legend = FALSE) + 
  geom_point(size = .2) + 
  ylim(c(0.4, 1)) +
  facet_grid(
    cols = vars(domain)
    # , rows = vars(what)
    , space = "free"
    , scales =  "free"
  ) + 
  labs(
    y = "F1", 
    x = "              No. training samples"
  ) + 
  theme(
    axis.title.y.left = element_text(angle = 0, vjust = .5)
    # , axis.text.x.bottom =element_text(angle = 35, hjust = 1, vjust = 1)
    , strip.clip = "off" # might requires ggplot2 dev version (see https://stackoverflow.com/a/73106205)
  ) 

p

(pn <- paste0(experiment_name, "_testset"))
cap <- paste(
  "Summary of test set performances of XLM-RoBERTa group mention detection classifiers in cross-lingual transfer experiment.",
  "The y-axis indicates performance in terms of the \\texttt{seqeval} F1 score of classifiers fine-tuned on annotated sentences UK parties' manifestos and then evaluated on, and successively adapted to sentences from German parties' manifestos.",
  "Points (line ranges) report the average ($\\pm$\\,1\\,std.\\,dev.) of performances of 5 different classifiers fine-tuned with different random seeds.",
  "Plot panels indicate whether the classifier was tested in English (the ``source'' language) or the German (the ``target'' language).",
  "For the ``German'' panel, the x-axis indicates whether the classifier fine-tuned on English-language manifestos was evaluated without adapting it to the target language (0, i.e., ``zero shot'')",
  "or, if not, how much labeled sentences were used to adapt the classifier (through continued training) befor evaluation.",
  # "Point shapes indicate evaluation scheme (i.e., different ways to compute the F1 score).",
  seqeval_note,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, "figure4b", cap = cap, w = 3, h = 2)


## Cross-domain transfer (from UK manifestos to UK parl. speech) ----

experiment_name <- "uk_cross-domain-transfer_deberta-finetuning"

fp <- file.path(experiment_results_path, experiment_name)

split_sizes <- read_tsv(file.path(fp, "split_sizes.tsv"))

test_res <- parse_test_jsonl(fp)

test_res <- transfer_test_results_to_df(test_res)

tmp <- split_sizes |> 
  filter(grepl("adapt", run_id)) |> 
  select(run_id, n_train) |> 
  separate(run_id, c("repetition", "fold"), sep = "-", remove = FALSE, extra = "merge") |> 
  distinct(fold, n_train) |> 
  mutate(n_train = cumsum(n_train)) |> 
  with(setNames(fold, n_train))

folds_map <- c(
  "test set" = "baseline", 
  "0" = "baseline-extra", 
  tmp
)
folds_map_inv <- set_names(names(folds_map), folds_map)

# NOTE: Figure 4(c)
p <- test_res |>
  filter(
    this %in% "SG" # c("micro", "SG")
    , what %in% "seqeval" # whats[1:2]
    , repetition %in% paste0("rep0", 1:3)
  ) |> 
  mutate(
    fold = factor(fold, folds_map, names(folds_map))
  ) |>
  group_by(what, this, domain, fold) |> 
  summarise(across(f1:recall, list(mean = mean, sd = sd))) |> 
  ggplot(
    aes(
      x = fold
      , y = f1_mean
      , ymin=f1_mean-f1_sd
      , ymax=f1_mean+f1_sd
    )
  )  + 
  geom_linerange(show.legend = FALSE) + 
  geom_point(pch = 21, fill = "white", show.legend = FALSE) + 
  geom_point(size = .2) + 
  ylim(c(0.4, 1)) +
  facet_grid(
    cols = vars(domain)
    # , rows = vars(what)
    , space = "free"
    , scales =  "free"
  ) + 
  labs(
    y = "F1", 
    x = "              No. training samples"
  ) + 
  theme(
    axis.title.y.left = element_text(angle = 0, vjust = .5)
    # , axis.text.x.bottom =element_text(angle = 35, hjust = 1, vjust = 1)
    , strip.clip = "off" # might requires ggplot2 dev version (see https://stackoverflow.com/a/73106205)
  ) 

p

(pn <- paste0(experiment_name, "_testset"))
cap <- paste(
  "Summary of test set performances of DeBERTa group mention detection classifiers in cross-domain transfer experiment.",
  "The y-axis indicates performance in terms of the \\texttt{seqeval} F1 score of classifiers fine-tuned on annotated manifesto sentences from UK party manifestos and then evaluated on, and successively adapted to sentences from parties' speeches in the UK House of Commons.",
  "Points (line ranges) report the average ($\\pm$\\,1\\,std.\\,dev.) of performances of 5 different classifiers fine-tuned with different random seeds.",
  "Plot panels indicate whether the classifier was tested in the source domain (i.e., party manifestos, left)",
  "or the target domain (i.e., parliamentary speech, right).",
  "For the ``target'' panel, the x-axis reports whether the classifier fine-tuned on party manifesto sentences was evaluated without adapting it to the target texts (0, i.e., ``zero shot'')",
  "or, if not, how much labeled sentences were used to adapt the classifier (through continued training) befor evaluation.",
  # "Point shapes indicate evaluation scheme (i.e., different ways to compute the F1 score).",
  seqeval_note,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, "figure4c", cap = cap, w = 3, h = 2)

