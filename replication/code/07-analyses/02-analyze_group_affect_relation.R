# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Application 2: Analyze relation between use of emotion words
#'          and group mentions in British party manifesto sentences
#' @author Ronja Sczepanski & Hauke Licht
#' @note   In this application, we show that sentences that contain mentions of 
#'          social groups are more likely to include emotional language than 
#'          sentences without group mentions.
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup -----

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(quanteda) # v3.2.3

library(sandwich)
library(lmtest)
library(clubSandwich)
library(broom)
library(texreg)
library(dsl) # v0.1.0

library(ggplot2)

data_path <- file.path("replication", "data")
utils_path <- file.path("replication", "code", "r_utils")
floats_path <- file.path("replication", "paper")
source(file.path(utils_path, "dsl_utils.R"))

# figure setup
source(file.path(utils_path, "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path = figs_dir)

# table setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)

# load manifesto data ----

## all unlabeled sentences ----

fp <- file.path(data_path, "corpora", "uk-manifesto_sentences.tsv")
man_dat <- read_tsv(fp, show_col_types = FALSE)

## all sentences with min. one mention ----

fp <- file.path(data_path, "labeled", "uk-manifesto_sentences_predicted_spans.tsv")
all_spans <- read_tsv(fp, show_col_types = FALSE)

## join info about which sentences where human-annotated ----

# NOTE: DSL approach (Egami et al. 2024) requires indiactor which sentence was human annotated
read_jsonl <- function(file) {
   map(
     read_lines(file), 
     jsonlite::fromJSON, 
     simplifyVector = FALSE
   )
}

fp <- file.path(data_path, "annotation", "labeled", "uk-manifestos_all_labeled.jsonl")
all_labeled <- read_jsonl(fp)

human_labeled_mentions <- tibble(value = all_labeled) |> 
  unnest_wider(value) |>
  transmute(
    sentence_id = map_chr(metadata, "sentence_id"),
    labels = labels |> map("BSCModel") |> map(unlist),
    # in the IOB2 scheme, label ID 6 = "B-social group"
    has_sg_span_silver = map_lgl(labels, ~6 %in% .x),
    n_sg_spans_silver = map_int(labels, ~sum(.x == 6)),
    labels = NULL
  )

## combine ----

dat_raw <- man_dat |>  
  select(manifesto_id, party, partyabbrev, sentence_id, text) |> 
  # add sentence-level mention counts
  left_join(
    all_spans |> 
      filter(label == "SG") |> 
      group_by(sentence_id) |> 
      summarise(
        n_sg_spans = n(),
        sg_spans = list(text)
      )
    , by = c("sentence_id")
  ) |>
  filter(!is.na(party))  |>
  mutate(n_sg_spans = replace_na(n_sg_spans, replace = 0L)) |> 
  left_join(
    human_labeled_mentions, 
    by = "sentence_id",
  )

unique(dat_raw$manifesto_id)

# some share of sentences has been distributed for annotation (below non-NA)
table(dat_raw$has_sg_span_silver, useNA = "always")

# apply the dictionary ----

fp <- file.path(data_path, "exdata", "liwc.rds")

if (!file.exists(fp)) {
  require(dataverse)
  # NOTE: we use the LIWC dictionary and get it from the replication 
  #        materials of Hargrave & Blumenau (2022)
  Sys.setenv("DATAVERSE_SERVER" = "dataverse.harvard.edu")
  Sys.setenv("DATAVERSE_ID" = "BJPolS")
  
  get_dataframe_by_name(
    filename = "liwc.Rdata",
    dataset = "doi:10.7910/DVN/PPSFLT",
    version = 1.0,
    original = TRUE,
    .f = function(x) load(x, envir = .GlobalEnv)
  )
  
  class(liwc)
  
  liwc <- liwc[c("Affect", "Posemo", "Negemo")]
  write_rds(liwc, fp)
} else {
  liwc <- read_rds(fp)
}

## apply to complete sentence text ----

corp <- corpus(select(dat_raw, sentence_id, text), text_field = "text", docid_field = "sentence_id")

toks <- corp |> 
  tokens(remove_punct = TRUE) |> 
  tokens_tolower()

n_toks <- tibble::enframe(ntoken(toks), name = "doc_id", value = "n_tokens")

dictionary_measures_df <- toks |> 
  tokens_lookup(dictionary = liwc) |> 
  dfm() |> 
  convert(to = "data.frame")

## robustness check: apply to sentences' text after removing text of any social group mentions ----

# NOTE: motivation here is that the group mentions might themselves contain words that are
#       in the LIWC dictionary, which could bias the results

dictionary_span_measures_df <- dat_raw |> 
  filter(n_sg_spans > 0) |> 
  select(sentence_id, sg_spans) |> 
  unnest_longer(sg_spans, indices_to = "span_nr") |> 
  mutate(span_id = sprintf("%s_%d", sentence_id, span_nr)) |> 
  corpus(text_field = "sg_spans", docid_field = "span_id") |> 
  tokens(remove_punct = TRUE) |> 
  tokens_tolower() |> 
  tokens_lookup(dictionary = liwc, valuetype = "glob") |> 
  dfm() |> 
  convert(to = "data.frame")

dictionary_span_measures_df <- dictionary_span_measures_df |> 
  tidyr::extract(doc_id, c("sentence_id", "span_nr"), regex = "^(.+)_(\\d+)$") |> 
  group_by(sentence_id) |> 
  summarise(across(c(affect, posemo, negemo), sum)) |> 
  ungroup()

dictionary_outside_span_measures_df <- dictionary_measures_df |> 
    pivot_longer(-1) |>
    left_join(
      pivot_longer(dictionary_span_measures_df, -1)
      , by = c("doc_id" = "sentence_id", "name")
      , suffix = c("", "_span")
    ) |> 
    mutate(
      value = ifelse(is.na(value_span), value, value - value_span),
      value_span = NULL
    ) |> 
    pivot_wider(names_glue = "{name}_outside_span")

# construct the data set ----

## code the dependent and independent variables  ----

dat <- dat_raw |> 
  select(-sg_spans) |> 
  # add sentence-level dictionary measures at 
  left_join(dictionary_measures_df, by = c("sentence_id" = "doc_id")) |> 
  # add sentence-level measures omitting spans at 
  left_join(dictionary_outside_span_measures_df, by = c("sentence_id" = "doc_id")) |> 
  # add number of tokens information
  left_join(n_toks, by = c("sentence_id" = "doc_id")) |> 
  mutate(
    group_mention = factor(n_sg_spans > 0, c(F, T), c("no", "yes")),
    group_mention_silver = factor(n_sg_spans_silver > 0, c(F, T), c("no", "yes")),
    # recode counter variables to binary indicators
    across(affect:negemo_outside_span, ~ifelse(. > 0, 1, 0))
  )

# inspect bivariate relation with predicted measures
prop.table(table("group mention" = dat$group_mention, "affective" = dat$affect), 1)
t.test(affect ~ group_mention, data = dat)

# predicted measures, only in set of human labeled sentences (i.e., "silver" labels)
with(
  subset(dat, !is.na(group_mention_silver)), 
  prop.table(table("group mention" = group_mention, "affective" = affect), 1)
)
t.test(affect ~ group_mention, data = subset(dat, !is.na(group_mention_silver)))
# note: magnitude of difference little smaller but relation holds

## human-labeled measures (in set of human labeled sentences)
with(
  subset(dat, !is.na(group_mention_silver)), 
  prop.table(table("group mention" = group_mention_silver, "affective" = affect), 1)
)
t.test(affect ~ group_mention_silver, data = subset(dat, !is.na(group_mention_silver)))
# note: magnitude of difference again a little smaller but relation holds

## add controls ----

### CMP party positions ----

fp <- file.path(data_path, "exdata", "cmp", "MPDataset_MPDS2023a.csv")
if (!file.exists(fp)) {
  url <- "https://manifesto-project.wzb.eu/down/data/2023a/datasets/MPDataset_MPDS2023a.csv"
  party_positions <- read_csv(url, show_col_types = FALSE)
  write_csv(party_positions, fp)
}

party_positions <- read_csv(fp, show_col_types = FALSE)
  


# subset to relevant country and elections
party_positions <- filter(party_positions, countryname == "United Kingdom", date >= 195910)

# compute manifesto economic and cultural left-right scores
party_positions$progressive_conservative <-  (party_positions$per104 + party_positions$per109 + party_positions$per601 + party_positions$per603 + party_positions$per605 + party_positions$per608) - (party_positions$per105 + party_positions$per106 + party_positions$per107 + party_positions$per501 + party_positions$per503 + party_positions$per602 + party_positions$per604 + party_positions$per607 + party_positions$per705)
party_positions$state_market<- (party_positions$per401 + party_positions$per402 + party_positions$per407 + party_positions$per414 + party_positions$per505) - (party_positions$per403 + party_positions$per404 + party_positions$per405 + party_positions$per406 + party_positions$per409 + party_positions$per412 + party_positions$per413 + party_positions$per415 + party_positions$per416 + party_positions$per504)

party_positions <- select(party_positions, -matches("^per\\d+"))

### Prime minister party indicator ----

# the date/edate information in the CMP record the date of the upcoming election
#  for which the coded manifesto was written.
# To know a parties pm_party status at the time of writing the manifesto,
#  we need to identify which party was in pm_party in the period leading up
#  to the upcoming election

election_date_mapping <- party_positions |> 
  distinct(date) |> 
  transmute(
    upcoming_election_date = date,
    ongoing_configuration = lag(date)
  )

configuration_pm_party_status <- party_positions |>
  select(party, partyname, ongoing_configuration = date, absseat, totseats, pervote) |> 
  group_by(ongoing_configuration) |>
  mutate(
    seats_share = absseat/totseats,
    pm_party = as.integer(absseat == max(absseat))
  ) |>
  select(-absseat, -totseats) |> 
  rename(votes_share = pervote) |> 
  left_join(election_date_mapping) |> 
  ungroup()

party_positions <- party_positions |> 
  select(-pervote, -absseat, totseats) |> 
  left_join(
    configuration_pm_party_status
    , by = c("party", "partyname", "date" = "upcoming_election_date")
  ) |> 
  as_tibble()

subset(party_positions, pm_party == 1, c(ongoing_configuration, edate, partyname, pm_party))
# note: this worked. while the Conservatives won the 2010 election, they 
#  were still in opposition when writing the manifesto 

party_positions <- party_positions |> 
  select(
    party, partyname,
    date, 
    progressive_conservative, 
    state_market, 
    ongoing_configuration,
    pm_party,
    votes_share,
    seats_share
  )

dat <- mutate(dat, date = sub("^(\\d+)_(\\d+)$", "\\2", manifesto_id, perl = TRUE))

date_mapping <- dat |> 
  distinct(date) |> 
  tidyr::separate(date, c("year", "month"), sep = "(?<=\\d{4})(?=\\d*)", remove = FALSE, extra = "merge") |> 
  left_join(
    party_positions |> 
      distinct(date) |> 
      tidyr::separate(date, c("year", "month"), sep = "(?<=\\d{4})(?=\\d*)", remove = FALSE, extra = "merge")
    , by = "year"
    , relationship = "many-to-many"
  ) |> 
  filter(month.x == "" | month.x == month.y) |> 
  select(date.x, date.y)


dat <- dat |> 
  left_join(date_mapping, by = c("date" = "date.x")) |> 
  left_join(party_positions, by = c("party", "date.y" = "date")) |> 
  select(-date) |> 
  rename(date = date.y) |> 
  mutate(
    # note for parties that entered, set PM party indcator to FALSE
    pm_party = replace_na(pm_party, FALSE)
  )
  
with(dat, table(group_mention, group_mention_silver, useNA = "ifany"))

# fit regressions ---- 

## regressions ----

# right-hand side of regression equation
rhs <- "group_mention + progressive_conservative + state_market + pm_party + n_tokens + as.factor(date)"

### only Lab and Con manifestos ----

tmpdat <- subset(dat, party %in% c(51620, 51320))

m1 <- glm(
  as.formula(paste("affect", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)
m1_pos <- glm(
  as.formula(paste("posemo", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)
m1_neg <- glm(
  as.formula(paste("negemo", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)

# get results with clustered standard errors (of type HC0)
m1_coefs <- coeftest(m1, vcov. = vcovCL(m1, cluster = tmpdat$manifesto_id, type = "HC0"))
m1_pos_coefs <- coeftest(m1_pos, vcov. = vcovCL(m1_pos, cluster = tmpdat$manifesto_id, type = "HC0"))
m1_neg_coefs <- coeftest(m1_neg, vcov. = vcovCL(m1_neg, cluster = tmpdat$manifesto_id, type = "HC0"))

m1_df <- broom::tidy(m1_coefs, conf.int = TRUE)
m1_pos_df <- broom::tidy(m1_pos_coefs, conf.int = TRUE)
m1_neg_df<- broom::tidy(m1_neg_coefs, conf.int = TRUE)

m1s <- bind_rows(
  "Emotions" = m1_df, 
  "Positive emotions" = m1_pos_df, 
  "Negative emotions" = m1_neg_df,
  .id = "model"
)

# log odds => odds
m1s |> 
  filter(term == "group_mentionyes") |>
  mutate(across(c(estimate, conf.low, conf.high), exp))

# NOTE: Figure 7
p <- m1s |> 
  filter(term == "group_mentionyes") |> 
  # use odds (instead of log odds)
  mutate(across(c(estimate, conf.low, conf.high), exp)) |> 
  ggplot(
    aes(
      y = reorder(model, desc(model)),
      x = estimate, 
      xmin = conf.low,
      xmax = conf.high
    )
  ) +
  geom_linerange() + 
  geom_point(pch = 21, fill = "white", show.legend = FALSE) + 
  geom_point(size = .2) + 
  geom_vline(xintercept = 1, linetype = 2, linewidth = 0.3) +
  # scale_shape()
  xlim(.95, 1.4) +
  labs(
    x = "Coefficient estimate (as odds)", 
    y = "",
  )

pn <- "regression_coefficients"
cap <- paste(
  "Coefficients estimates from logistic regressions analyzing whether sentences that contain group mentions are more likely to contain emotion words.",
  "The x-axis reports our estimates of the odds that a sentence contains emotional language when it contains at least one social group mention compared to when it contains no social group mention.",
  "Points (line ranges) report the coefficients point estimates (95\\% confidence intervals) of logistic regression models.",
  # "Point shapes differentiate between different dictionary-based outcomes (i.e., considering all affect words or only positive/negative affect words).",
  # "The y-axis differentiates between regression models fitted to different subsets of our British party manifesto corpus.",
  "The y-axis values differentiate between different emotion dictionary categories.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, "figure7", cap = cap, w = 4.5, h = 1.5)

# NOTE: Table H1
texreg(
  l = list(
    "Emotions" = m1, 
    "Positive emotions" = m1_pos, 
    "Negative emotions" = m1_neg
  )
  , caption = paste(
    "Logistic regression coefficient estimates", 
    "from regressing binary sentence-level indicator of the use of LIWC emotion words (positive or negative/positive/negative) in the sentence", 
    "on indicator for whether the sentence is predicted to mention at least one social group by our RoBERTa group mention detection classifier",
    "in the Conservative and Labour party manifestos in our UK corpus.",
    collapse = " "
  ) 
  , custom.note = paste(
    "\\item %stars.",
    "\\item All models include election fixed effects.",
    "\\item Standard errors clustered by party and election."
  )
  , label = "tab:regression_coefficients"
  , file = file.path(tabs_dir, "tableH1.tex")
  , override.coef = list(
    "Emotions" = m1_df$estimate, 
    "Positive emotions" = m1_pos_df$estimate, 
    "Negative emotions" = m1_neg_df$estimate
  )
  , override.se = list(
    "Emotions" = m1_df$std.error, 
    "Positive emotions" = m1_pos_df$std.error, 
    "Negative emotions" = m1_neg_df$std.error
  )
  , override.pvalues = list(
    "Emotions" = m1_df$p.value, 
    "Positive emotions" = m1_pos_df$p.value, 
    "Negative emotions" = m1_neg_df$p.value
  )
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = list(
    "group_mentionyes" = "Contains social group mention(s)",
    "progressive_conservative" = "Progressive--conservative position",
    "state_market" = "State--market position",
    "pm_party" = "Prime minister party",
    "n_tokens" = "$N$ tokens"
  )
  , leading.zero = TRUE
  , single.row = FALSE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
) 


#### robustness: dictionary measures applied only to text without words belonging to social group mentions (Table H3) ----

dat |> with(table(affect, affect_outside_span))
tmpdat <- filter(dat, party %in% c(51620, 51320), !(affect == 1 & affect_outside_span == 0))

m1r <- glm(
  as.formula(paste("affect", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)
m1r_pos <- glm(
  as.formula(paste("posemo", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)
m1r_neg <- glm(
  as.formula(paste("negemo", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)

# get results with clustered standard errors (of type HC0)
m1r_coefs <- coeftest(m1r, vcov. = vcovCL(m1r, cluster = tmpdat$manifesto_id, type = "HC0"))
m1r_pos_coefs <- coeftest(m1r_pos, vcov. = vcovCL(m1r_pos, cluster = tmpdat$manifesto_id, type = "HC0"))
m1r_neg_coefs <- coeftest(m1r_neg, vcov. = vcovCL(m1r_neg, cluster = tmpdat$manifesto_id, type = "HC0"))

m1r_df <- broom::tidy(m1r_coefs, conf.int = TRUE)
m1r_pos_df <- broom::tidy(m1r_pos_coefs, conf.int = TRUE)
m1r_neg_df<- broom::tidy(m1r_neg_coefs, conf.int = TRUE)

m1rs <- bind_rows(
  "Emotions" = m1r_df, 
  "Positive emotions" = m1r_pos_df, 
  "Negative emotions" = m1r_neg_df,
  .id = "model"
)

m1rs |>  filter(term == "group_mentionyes")

# log odds => odds
m1rs |> 
  filter(term == "group_mentionyes") |>
  mutate(across(c(estimate, conf.low, conf.high), exp))

# NOTE: Table H3
texreg(
  l = list(
    "Emotions" = m1r,
    "Positive emotions" = m1r_pos,
    "Negative emotions" = m1r_neg
  )
  , caption = paste(
    "Logistic regression coefficient estimates", 
    "from regressing binary sentence-level indicator of the use of LIWC emotion words (positive or negative/positive/negative) in the sentence", 
    "on indicator for whether the sentence is predicted to mention at least one social group by our RoBERTa group mention detection classifier",
    "in the Conservative and Labour party manifestos in our UK corpus.",
    "As a robustness check, these models exclude sentences in which \\emph{all} emotion words detected with dictionary are located in the predicted group mention(s).",
    collapse = " "
  )
  , custom.note = paste(
    "\\item %stars.",
    "\\item All models include election fixed effects.",
    "\\item Standard errors clustered by party and election."
  )
  , label = "tab:regression_coefficients_subset"
  , file = file.path(tabs_dir, "tableH3.tex")
  , override.coef = list(
    "Emotions" = m1r_df$estimate,
    "Positive emotions" = m1r_pos_df$estimate,
    "Negative emotions" = m1r_neg_df$estimate
  )
  , override.se = list(
    "Emotions" = m1r_df$std.error,
    "Positive emotions" = m1r_pos_df$std.error,
    "Negative emotions" = m1r_neg_df$std.error
  )
  , override.pvalues = list(
    "Emotions" = m1r_df$p.value,
    "Positive emotions" = m1r_pos_df$p.value,
    "Negative emotions" = m1r_neg_df$p.value
  )
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = list(
    "group_mentionyes" = "Contains social group mention(s)",
    "progressive_conservative" = "Progressive--conservative position",
    "state_market" = "State--market position",
    "pm_party" = "Prime minister party",
    "n_tokens" = "$N$ tokens"
  )
  , leading.zero = TRUE
  , single.row = FALSE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
)

### all parties (Table H4) ----

rhs <- "group_mention + progressive_conservative + state_market + pm_party + n_tokens + as.factor(date)"

m2 <- glm(
  as.formula(paste("affect", rhs, sep = "~")),
  data = dat, 
  family = binomial(link = "logit")
)
m2_pos <- glm(
  as.formula(paste("posemo", rhs, sep = "~")),
  data = dat,
  family = binomial(link = "logit")
)
m2_neg <- glm(
  as.formula(paste("negemo", rhs, sep = "~")),
  data = dat,
  family = binomial(link = "logit")
)

# get results with clustered standard errors (of type HC0)
m2_coefs <- coeftest(m2, vcov. = vcovCL(m2, cluster = dat$manifesto_id, type = "HC0"))
m2_pos_coefs <- coeftest(m2_pos, vcov. = vcovCL(m2_pos, cluster = dat$manifesto_id, type = "HC0"))
m2_neg_coefs <- coeftest(m2_neg, vcov. = vcovCL(m2_neg, cluster = dat$manifesto_id, type = "HC0"))

m2_df <- broom::tidy(m2_coefs, conf.int = TRUE)
m2_pos_df <- broom::tidy(m2_pos_coefs, conf.int = TRUE)
m2_neg_df<- broom::tidy(m2_neg_coefs, conf.int = TRUE)

m2s <- bind_rows(
  "Emotions" = m2_df, 
  "Positive emotions" = m2_pos_df, 
  "Negative emotions" = m2_neg_df,
  .id = "model"
)

# log odds => odds
m2s |> 
  filter(term == "group_mentionyes") |>
  mutate(across(c(estimate, conf.low, conf.high), exp))

# NOTE: Table H4
texreg(
  l = list(
    "Emotions" = m2, 
    "Positive emotions" = m2_pos, 
    "Negative emotions" = m2_neg
  )
  , caption = paste(
    "Logistic regression coefficient estimates", 
    "from regressing binary sentence-level indicator of the use of LIWC emotion words (positive or negative/positive/negative) in the sentence", 
    "on indicator for whether the sentence is predicted to mention at least one social group by our RoBERTa group mention detection classifier",
    "in all party manifestos in our UK corpus.",
    collapse = " "
  ) 
  , custom.note = paste(
    "\\item %stars.",
    "\\item All models include election fixed effects.",
    "\\item Standard errors clustered by party and election."
  )
  , label = "tab:regression_coefficients_all_parties"
  , file = file.path(tabs_dir, "tableH4.tex")
  , override.coef = list(
    "Emotions" = m2_df$estimate, 
    "Positive emotions" = m2_pos_df$estimate, 
    "Negative emotions" = m2_neg_df$estimate
  )
  , override.se = list(
    "Emotions" = m2_df$std.error, 
    "Positive emotions" = m2_pos_df$std.error, 
    "Negative emotions" = m2_neg_df$std.error
  )
  , override.pvalues = list(
    "Emotions" = m2_df$p.value, 
    "Positive emotions" = m2_pos_df$p.value, 
    "Negative emotions" = m2_neg_df$p.value
  )
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = list(
    "group_mentionyes" = "Contains social group mention(s)",
    "progressive_conservative" = "Progressive--conservative position",
    "state_market" = "State--market position",
    "pm_party" = "Prime minister party",
    "n_tokens" = "$N$ tokens"
  )
  , leading.zero = TRUE
  , single.row = FALSE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
) 


### all parties 2015 onward (Table H5) ----

tmpdat <- subset(dat, date >= 201505)

m3 <- glm(
  as.formula(paste("affect", rhs, sep = "~")),
  data = tmpdat, 
  family = binomial(link = "logit")
)
m3_pos <- glm(
  as.formula(paste("posemo", rhs, sep = "~")),
  data = tmpdat,
  family = binomial(link = "logit")
)
m3_neg <- glm(
  as.formula(paste("negemo", rhs, sep = "~")),
  data =  tmpdat,
  family = binomial(link = "logit")
)

# get results with clustered standard errors (of type HC0)
m3_coefs <- coeftest(m3, vcov. = vcovCL(m3, cluster = tmpdat$manifesto_id, type = "HC0"))
m3_pos_coefs <- coeftest(m3_pos, vcov. = vcovCL(m3_pos, cluster = tmpdat$manifesto_id, type = "HC0"))
m3_neg_coefs <- coeftest(m3_neg, vcov. = vcovCL(m3_neg, cluster = tmpdat$manifesto_id, type = "HC0"))

m3_df <- broom::tidy(m3_coefs, conf.int = TRUE)
m3_pos_df <- broom::tidy(m3_pos_coefs, conf.int = TRUE)
m3_neg_df<- broom::tidy(m3_neg_coefs, conf.int = TRUE)

m3s <- bind_rows(
  "Emotions" = m3_df, 
  "Positive emotions" = m3_pos_df, 
  "Negative emotions" = m3_neg_df,
  .id = "model"
)

# log odds => odds
m3s |> 
  filter(term == "group_mentionyes") |>
  mutate(across(c(estimate, conf.low, conf.high), exp))

# NOTE: Table H5
texreg(
  l = list(
    "Emotions" = m3, 
    "Positive emotions" = m3_pos, 
    "Negative emotions" = m3_neg
  )
  , caption = paste(
    "Logistic regression coefficient estimates", 
    "from regressing binary sentence-level indicator of the use of LIWC emotion words (positive or negative/positive/negative) in the sentence", 
    "on indicator for whether the sentence is predicted to mention at least one social group by our RoBERTa group mention detection classifier",
    "in all party manfistos from elections 2015 onwards in our UK corpus.",
    collapse = " "
  ) 
  , custom.note = paste(
    "\\item %stars.",
    "\\item All models include election fixed effects.",
    "\\item Standard errors clustered by party and election."
  )
  , label = "tab:regression_coefficients_2015onwards"
  , file = file.path(tabs_dir, "tableH5.tex")
  , override.coef = list(
    "Emotions" = m3_df$estimate, 
    "Positive emotions" = m3_pos_df$estimate, 
    "Negative emotions" = m3_neg_df$estimate
  )
  , override.se = list(
    "Emotions" = m3_df$std.error, 
    "Positive emotions" = m3_pos_df$std.error, 
    "Negative emotions" = m3_neg_df$std.error
  )
  , override.pvalues = list(
    "Emotions" = m3_df$p.value, 
    "Positive emotions" = m3_pos_df$p.value, 
    "Negative emotions" = m3_neg_df$p.value
  )
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = list(
    "group_mentionyes" = "Contains social group mention(s)",
    "progressive_conservative" = "Progressive--conservative position",
    "state_market" = "State--market position",
    "pm_party" = "Prime minister party",
    "n_tokens" = "$N$ tokens"
  )
  , leading.zero = TRUE
  , single.row = FALSE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
) 

## DSL estimates ----

# renv::install("naoki-egami/dsl") 
# - requires cmake (e.g. via brew install --cask cmake, result of `$(which cmake)` must be in PATH)
# - requires {RcppEigen} ≥ 0.3.3.9.4 "for {lme4}"

### only Lab and Con manifestos ----

tmpdat <- as.data.frame(subset(dat, party %in% c(51620, 51320)))

with(tmpdat, table(group_mention, group_mention_silver, useNA = "always"))

table(tmpdat$group_mention, useNA = "always")
tmpdat$group_mention <- tmpdat$group_mention == "yes"
table(tmpdat$group_mention_silver, useNA = "always")
tmpdat$group_mention_silver <- tmpdat$group_mention_silver == "yes"
table(tmpdat$group_mention_silver, useNA = "always")

# flag human-labeled samples
tmpdat$labeled_ <- as.integer(!is.na(tmpdat$group_mention_silver))
table(tmpdat$labeled_)

rhs <- "group_mention_silver + progressive_conservative + state_market + pm_party + n_tokens + as.factor(date)"
groundtruth_var <- "group_mention_silver"
pred_var <- "group_mention"

m1_dsl <- dsl(
  model = "logit",
  formula = as.formula(paste("affect", rhs, sep = "~")),
  predicted_var = groundtruth_var, # the variable that got predicted
  prediction = pred_var, # the variable that records the predicted values
  labeled = "labeled_",
  data = tmpdat,
  cluster = "manifesto_id",
  cross_fit = 5, # default value
  sample_split = 10, # default value
  seed = 1234
)
m1_dsl_pos <- dsl(
  model = "logit",
  formula = as.formula(paste("posemo", rhs, sep = "~")),
  predicted_var = groundtruth_var, # the variable that got predicted
  prediction = pred_var, # the variable that records the predicted values
  labeled = "labeled_",
  data = tmpdat,
  cluster = "manifesto_id",
  cross_fit = 5, # default value
  sample_split = 10, # default value
  seed = 1234
)
m1_dsl_neg <- dsl(
  model = "logit",
  formula = as.formula(paste("negemo", rhs, sep = "~")),
  predicted_var = groundtruth_var, # the variable that got predicted
  prediction = pred_var, # the variable that records the predicted values
  labeled = "labeled_",
  data = tmpdat,
  cluster = "manifesto_id",
  cross_fit = 5, # default value
  sample_split = 10, # default value
  seed = 1234
)

m1_dsl_df <- tidy(m1_dsl, conf.int = TRUE)
m1_dsl_pos_df <- tidy(m1_dsl_pos, conf.int = TRUE)
m1_dsl_neg_df <- tidy(m1_dsl_neg, conf.int = TRUE)

m1s_dsl <- bind_rows(
  "Emotions" = m1_dsl_df, 
  "Positive emotions" = m1_dsl_pos_df, 
  "Negative emotions" = m1_dsl_neg_df,
  .id = "model"
)

filter(m1s, term == "group_mentionyes")
filter(m1s_dsl, term == "group_mention_silverTRUE")

# NOTE: Table H2
texreg(
  l = list(
    "Emotions" = m1_dsl,
    "Positive emotions" = m1_dsl_pos,
    "Negative emotions" = m1_dsl_neg
  )
  , caption = paste(
    "Adjusted coefficient estimates", 
    "obtained with design-based supervised learning method propsed by \\citet{egami_using_2024}",
    "for logistic regression regressing binary sentence-level indicator of the use of LIWC emotion words (positive or negative/positive/negative) in the sentence", 
    "on indicator for whether the sentence is predicted to mention at least one social group by our RoBERTa group mention detection classifier",
    "in the Conservative and Labour party manifestos in our UK corpus.",
    collapse = " "
  ) 
  , custom.note = paste(
    "\\item %stars.",
    "\\item All models include election fixed effects.",
    "\\item Standard errors clustered by party and election."
  )
  , label = "tab:regression_coefficients_dsl"
  , file = file.path(tabs_dir, "tableH2.tex")
  , override.coef = list(
    "Emotions" = m1_dsl_df$estimate,
    "Positive emotions" = m1_dsl_pos_df$estimate,
    "Negative emotions" = m1_dsl_neg_df$estimate
  )
  , override.se = list(
    "Emotions" = m1_dsl_df$std.error,
    "Positive emotions" = m1_dsl_pos_df$std.error,
    "Negative emotions" = m1_dsl_neg_df$std.error
  )
  , override.pvalues = list(
    "Emotions" = m1_dsl_df$p.value,
    "Positive emotions" = m1_dsl_pos_df$p.value,
    "Negative emotions" = m1_dsl_neg_df$p.value
  )
  , stars = c(0.001, 0.01, 0.05)
  , custom.coef.map = list(
    "group_mention_silverTRUE" = "Contains social group mention(s)",
    "progressive_conservative" = "Progressive--conservative position",
    "state_market" = "State--market position",
    "pm_party" = "Prime minister party",
    "n_tokens" = "$N$ tokens"
  )
  , leading.zero = TRUE
  , single.row = FALSE
  , caption.above = TRUE
  , center = TRUE
  , digits = 3
  , dcolumn = TRUE
  , threeparttable = TRUE
  , booktabs = TRUE
  , use.packages = FALSE
) 

