# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Describe annotations and inter-coder agreement
#' @author Hauke Licht
#' @date   2023-04-17
#' @note   Spans extracted with Roberta (base) token classifier trained on
#'          labels aggregated from RA's annotations from all UK manifestos
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)

# TODO: adjust path
data_path <- file.path("replication", "data")
floats_path <- file.path("replication", "paper")
utils_path <- file.path("replication", "code", "r_utils")

# tables setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)

# load annotations for all jobs ----

parse_json_line <- function(line) {
  content <- jsonlite::fromJSON(line, simplifyDataFrame = FALSE, flatten = FALSE)
  content$tokens <- list(content$tokens)
  content$annotations <- list(content$annotations)
  content$labels <- list(content$labels)
  return(as_tibble(content))
}

parse_annotations_jsonl_file <- function(fp) {
  lines <- read_lines(fp)
  out <- map_dfr(lines, parse_json_line)
  out$labels <- map_if(out$labels, ~!is.null(.), "GOLD")
  return(out)
}

fps <- list.files(
  file.path(data_path, "annotation", "parsed"), 
  pattern = "_annotations\\.jsonl$", 
  full.names = TRUE,
  recursive = TRUE
)

worker_responses <- map_dfr(fps, parse_annotations_jsonl_file)

fps <- list.files(
  file.path(data_path, "annotation", "annotations"),
  pattern = "_ids\\.csv$", 
  full.names = TRUE,
  recursive = TRUE
)

# set job names
names(fps) <- job_ids <- basename(dirname(fps))

sentence_ids <- map_dfr(fps, read_csv, .id = "job_id", show_col_types = FALSE)
sentence_ids <- select(sentence_ids, job_id, sentence_id, uid)

worker_responses <- left_join(worker_responses, sentence_ids, by = c("id" = "uid"))


# ensure all are not NA
table(is.na(worker_responses$sentence_id))
table(is.na(worker_responses$job_id))

worker_responses <- arrange(worker_responses, job_id, id)

# unnest
worker_responses <- unnest_longer(worker_responses, col=annotations, values_to="annotations", indices_to="annotator")

# discard B-/I- mention distinction (not necessary for eval)
worker_responses$annotations <- map(worker_responses$annotations, ~ifelse(.>6, .-6, .))

# parse
annotations <- worker_responses %>% 
  group_by(job_id, id, text) %>% 
  summarise(
    tokens = list(tokens[[1]])
    , n_annotators =  n_distinct(annotator)
    , annotators = list(c(annotator))
    , annotations = list(do.call(rbind, annotations))
  ) %>% 
  ungroup() 

# samples summary ----

samples_sum <- count(sentence_ids, job_id) %>%
  mutate(
    description = case_when(
      grepl("uk-manifestos-round-0", job_id) ~ "1 UK: Labour and Conservative manifestos (1964-2015)",
      grepl("uk-manifestos-round", job_id) ~ "2 UK: Labour and Conservative manifestos (2017 and 2019)",
      grepl("uk-manifestos-other", job_id) ~ "3 UK: DUP, Greens, LibDem, SNP, and UKIP (2015-2019)",
      grepl("uk-commons", job_id) ~ "4 UK: House of Commons speeches (2013-2019)",
      job_id == "de-manifestos-round-01" ~ "5 Germany: CDU and SPD manifestos (2002-2021)",
      job_id == "de-manifestos-round-02" ~ "6 Germany: AfD, B90/GRÜNE, FDP, and LINKE (2013-2021)"
    )
  ) %>% 
  separate(description, c("nr", "description"), sep = " ", extra = "merge") %>% 
  arrange(nr) %>% 
  select(nr, description, job_id, n)

# NOTE: Table B1
samples_sum %>% 
  select(-job_id) %>% 
  group_by(nr, description) %>% 
  summarise(n = sum(n)) %>% 
  ungroup() %>% 
  select(-nr) %>% 
  quick_kable(
    caption = "Number of sentences distributed for annotation per annotation job."
    , col.names = c("Annotation job", "$N$")
    , align = c("l", "r")
    , label = "n_samples_per_job"
  ) %>% 
  save_kable(.file.name = "tableB1")

# annotations jobs summaries ----

job_sum <- annotations %>% 
  group_by(job_id, n_annotators) %>% 
  summarise(n = n_distinct(id)) %>% 
  ungroup() %>% 
  mutate(n_annotators = ifelse(n_annotators == 1, "singly", "doubly")) %>% 
  pivot_wider(names_from = n_annotators, values_from = n) %>% 
  mutate(
    description = case_when(
      grepl("uk-manifestos-round", job_id) ~ "1 UK: Labour and Conservative manifestos (1964-2019)",
      grepl("uk-manifestos-other", job_id) ~ "3 UK: DUP, Greens, LibDem, SNP, and UKIP (2015-2019)",
      grepl("uk-commons", job_id) ~ "4 UK: House of Commons speeches (2013-2019)",
      job_id == "de-manifestos-round-01" ~ "5 Germany: CDU and SPD manifestos (2002-2021)",
      job_id == "de-manifestos-round-02" ~ "6 Germany: AfD, B90/GRÜNE, FDP, and LINKE (2013-2021)"
    )
  ) %>% 
  separate(description, c("nr", "description"), sep = " ", extra = "merge") %>% 
  arrange(nr) 

# NOTE: Table B2
job_sum %>% 
  select(-job_id) %>% 
  group_by(nr, `Annotation job` = description) %>%
  summarise(`2` = sum(doubly), `1` = sum(singly)) %>%
  # mutate(prop = `2`/(`1`+`2`))
  ungroup() %>% 
  select(-nr) %>% 
  quick_kable(
    caption = "Number of sentences annotated by two or one coder per dataset."
    , align = c("l", "r", "r")
    , label = "n_annotations_per_job"
  ) %>% 
  add_header_above(c(" " = 1, "$N$ coders" = 2), escape = FALSE) %>% 
  save_kable(.file.name = "tableB2")

# annotation summaries ----

annotations_stats <- annotations %>% 
  mutate(
    # how many tokens highlighted (cross-annotator average)
    prop_unlabeled = map_dbl(annotations, ~mean(. == 0))
    # none highlighted 
    , all_unlabeled = prop_unlabeled == 1.0
    # any "social group" annotations 
    , any_social_groups = map_lgl(annotations, ~any(. == 1))
  ) %>% 
  mutate(
    cat_ = case_when(
      all_unlabeled ~ "no mention at all"
      , !any_social_groups ~ "no social group mention"
      , TRUE ~ "min. one social group mention"
    ),
    cat_ = factor(cat_, c("no mention at all", "no social group mention", "min. one social group mention"))
  ) 

# NOTE: Table B4
annotations_stats %>% 
  left_join(select(job_sum, nr, description, job_id)) %>% 
  count(nr, description, cat_) %>% 
  group_by(nr, description) %>% 
  mutate(value = sprintf("%d (%0.02f)", n, n/sum(n))) %>% 
  ungroup() %>% 
  select(-n) %>% 
  rename(`Annotation job` = description) %>% 
  pivot_wider(names_from = cat_) %>% 
  arrange(nr) %>% 
  select(-nr) %>% 
  quick_kable(
    caption = paste(
      "Proportions of sentences by annotation job that contain",
      "(i) no mention at all,",
      "(ii) no social group mention (but min. one mention of another group type), or",
      "(iii) at least one social group mention.",
      collapse = " "
    ),
    label = "prop_mentions_by_job",
    align = c("l", rep("r", 3))
  ) %>% 
  save_kable(.file.name = "tableB4")

# compute inter-coder agreement metrics ----

compute_metrics <- function(x) {
  out <- list()
  out$all <- irr::agree(t(x))$value
  # binary (using "social group" as target label)
  out$binary <- irr::agree(t(matrix(as.numeric(x==1), ncol = ncol(x))))$value
  return(out)
}

ira_metrics <- annotations_stats %>% 
  filter(n_annotators > 1) %>% 
  mutate(agreement = map(annotations, compute_metrics)) %>% 
  unnest_wider(agreement, names_sep = "_")


# NOTE: Table B5
ira_metrics %>% 
  left_join(select(job_sum, nr, description, job_id)) %>% 
  filter(!all_unlabeled) %>% 
  group_by(any_social_groups, nr, description) %>% 
  summarise(
    n = n(), 
    across(starts_with("agreement"), list(q10 = ~quantile(. ,.25), mean = mean, median = median))
  ) %>% 
  ungroup() %>% 
  arrange(desc(any_social_groups), nr) %>% 
  select(-nr) %>% 
  mutate(any_social_groups = factor(any_social_groups, c(F,T), c("no social group mention", "min. one social group mention"))) %>% 
  mutate(across(-c(1:3), ~sprintf("%.02f", ./100))) %>% 
  quick_kable(
    caption = paste(
      "Summary statistics of sentence-level inter-coder agreement scores by annotation job in doubly annotated sentences that are coded by at least one coder as containing at least one group mention annotation.",
      "The top panel reports agreement when counting only group mention annotations (and treating all other annotations as outside a span).",
      "The bottom panel, in contrast, reports agreement when considering all five group categories in our coding scheme.",
      "\\emph{Note:} Sentence with no annotation by either coder omitted because agreement is 100\\% in all.",
      collapse = " "
    ),
    label = "intercoder_agreement",
    col.names = c("tmp", "Annotation job", "$N$", rep(c("10\\% ptl.", "Mean", "Median"), 2)),
    align = c(rep("l", 2), rep("r", 7))
  ) %>% 
  add_header_above(c(" " = 3, "All group categories" = 3, "Social group vs. none (binary)" = 3)) %>% 
  collapse_rows(1:2, latex_hline = "major", row_group_label_position = "stack") %>% 
  save_kable(.file.name = "tableB5")


# analyze (dis)agreement patterns in those with low (binary) K alpha ----

analyze_agreement_patterns <- function(x) { # 
  
  # set non-social group annotations to none
  x[x != 1] <- 0L
  
  # compute agreement values 
  kas <- list(
    "all" = irr::agree(t(x)),
    "those annotated" = irr::agree(t(as.matrix(x[, colMeans(x) > 0])))
  )
  
  idxs <- which(colMeans(x) > 0)
  if ((l <- length(idxs)) > 1) {
    x <- split(as.data.frame(t(x[, idxs])), cumsum(c(T, idxs[-l] != idxs[-1]-1L)))
    x <- unname(lapply(x, compose(unname, as.matrix, t)))
  } else {
    x <- list(as.matrix(x[, idxs]))
  }
  
  # analyze agreement patterns
  aps <- list()
  for (i in seq_along(x)) { # i <- 1L
    # where column mean is 1, there is agreement
    a <- colMeans(x[[i]]) == 1
    if (all(a)) {
      aps[[i]] <- "complete agreement"
    } 
    else if (all(!a)) {
      aps[[i]] <- "no agreement"
    } 
    else if (max(which(a)) < min(which(!a))) {
      aps[[i]] <- "agreement on first part"
    }
    else if (min(which(a)) > max(which(!a)))  {
      aps[[i]] <- "agreement on last part"
    }
    else if (!any( which(!a) %in% seq(min(which(a)), max(which(a))) ) )  {
      aps[[i]] <- "agreement on mid part"
    }
    else {
      aps[[i]] <- "other agreement pattern"
    }
  }
  
  return(list("kas" = kas, "annotations" = x, "patterns" = aps))
}

agreement_patterns <- ira_metrics %>%
  filter(!all_unlabeled) %>%
  filter(any_social_groups) %>%
  mutate(agreement_patterns = map(annotations, analyze_agreement_patterns))

# NOTE: Table B6
agreement_patterns %>%
  left_join(select(job_sum, nr, description, job_id)) %>%
  select(nr, description, job_id, id, agreement_patterns) %>%
  unnest_wider(agreement_patterns) %>%
  select(nr, description, job_id, id, patterns) %>%
  unnest_longer(patterns) %>%
  filter(patterns != "complete agreement") %>%
  count(nr, description, patterns) %>% 
  group_by(nr, description) %>% 
  mutate(
    prop = n/sum(n)
    , n = sum(n)
    , patterns = factor(
      patterns
      , c("no agreement", "agreement on first part", "agreement on mid part", "agreement on last part",  "other agreement pattern")
      , c("none", "on first part", "on mid part", "on last part",  "other pattern")
    )
  ) %>% 
  arrange(desc(prop)) %>% 
  ungroup() %>% 
  pivot_wider(names_from = patterns, values_from = prop) %>% 
  arrange(nr) %>% 
  select(
    `Annotation job` = description, 
    `$N$` = n, 
    !!c("none", "on first part", "on mid part", "on last part",  "other pattern")
  ) %>% 
  quick_kable(
    caption = "Distribution of disagreement patterns in sentences segments with intercoder disagreement by annotation job."
    , align = c("l", rep("r", 6))
    , label = "agreement_patterns"
  ) %>% 
  add_header_above(c(" " = 2, "Agreement" = 5)) %>% 
  save_kable(.file.name = "tableB6")

