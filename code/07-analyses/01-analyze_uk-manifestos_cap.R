# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Application 1: What distinguishes British parties’ social group focus
#' @author Hauke Licht
#' @note:  In application 1 (section 5.1), we study differences in British 
#'          parties' social group focus regarding (a) how much they emphasize 
#'          groups in different policy areas and (b) what distinguishes the 
#'          groups they mention
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(quanteda) # v3.2.3

# TODO: update paths
utils_path <- file.path("replication", "code", "r_utils")
data_path <- file.path("replication", "data")
floats_path <- file.path("replication", "paper")

source(file.path(utils_path, "figthing_words.R")) # for `tokenize.corpus()` and `textstat_fighting_words()`

# figure setup
source(file.path(utils_path, "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path = figs_dir)

# table setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)


types <- c("SG", "PG", "PI", "ORG", "ISG")

parties_colors <- read_tsv(file.path(data_path, "uk_parties.tsv"))
parties_names_map <-  with(parties_colors, set_names(party_name_short, party_name))
parties_short_names_map <- with(parties_colors, set_names(party_name_short, party_name))
parties_colors_map <- with(parties_colors, set_names(color_dark, party_name))
parties_short_colors_map <- with(parties_colors, set_names(color_dark, party_name_short))

# load our span- and sentence-level measurements ----

fp <- file.path(data_path, "corpora", "uk-manifesto_sentences.tsv")
man_dat <- read_tsv(fp, show_col_types = FALSE)
parties <- distinct(man_dat, party, partyname)
party2id <- with(parties, setNames(party, partyname))
id2party <- with(parties, setNames(partyname, party))

fp <- file.path(data_path, "labeled", "uk-manifesto_sentences_predicted_spans.tsv")
all_spans <- read_tsv(fp, show_col_types = FALSE)


# Salience by party (Lab & Con) over time ----

#' Note: we omly report time-series for Labour and Conservatives because for 
#'  other parties in UK and DE, we have only relatively few manifestos

tmp <- all_spans |> 
  group_by(across(1:4)) |> 
  summarise(
    n_spans = n()
    , n_sg_spans = sum(label == "SG")
  ) |> 
  right_join(
    select(man_dat, partyname, sentence_id, text)
    , by = c("sentence_id")
  ) |> 
  filter(!is.na(party))

tmp$n_spans[is.na(tmp$n_spans)] <- 0L
tmp$n_sg_spans[is.na(tmp$n_sg_spans)] <- 0L

set.seed(1234)
p_dat <- tmp |> 
  group_by(manifesto_id, date, party, partyname) |> 
  summarise(
    # bootstrap salience estimate
    bs = list(replicate(100, mean(sample(n_sg_spans > 0, replace = TRUE))))
  ) |> 
  mutate(
    q10 = map_dbl(bs, quantile, 0.1)
    , mean = map_dbl(bs, mean)
    , q90 = map_dbl(bs, quantile, 0.9)
  ) |> 
  ungroup()

# Figure F2
p <- p_dat |> 
  filter(party %in% c(51320, 51620)) |>
  mutate(
    election = case_when(
      date == 201706 ~ "2017",
      date == 201912 ~ "2019",
      nchar(date) > 4 ~ sub("(?<=\\d{4})(?=\\d{2})", "-", date, perl = TRUE),
      TRUE ~ as.character(date)
    )
  ) |> 
  arrange(partyname, election) |> 
  ggplot(
    aes(
      x = factor(election)
      , y = mean
      , ymin = q10
      , ymax = q90
      , color = partyname
      , group = partyname
    )
  ) + 
  geom_path(position = position_dodge(.1) ,show.legend = FALSE) +
  geom_linerange(position = position_dodge(.1) ,show.legend = FALSE) +
  geom_point(position = position_dodge(.1), fill = "white", pch = 21, size = 2, show.legend = FALSE) + 
  geom_point(position = position_dodge(.1), size = 0.5) +
  scale_color_manual(breaks = names(parties_colors_map)[c(1, 4)], values = parties_colors_map[c(1, 4)]) + 
  guides(
    color = guide_legend(title = NULL, override.aes = list(shape = 15, size = 10/.pt))
  ) + 
  labs(
    y = "Share of sentences with at least one social group mention"
    , x = "Election"
  ) + 
  theme(
    legend.position = "top"
    , axis.text.x.bottom = element_text(angle = 33, hjust=1, vjust = 1)
  )

p
pn <- "lab_con_sg_salience_overtime"
cap <- paste(
  "Share of sentences in Labour and Conservative Party manifestos that contain at least one social group mention by election.",
  "\\emph{Note:} To quantify the uncertainty in these estimates, we have bootstrapped sentence-level indicators.",
  "Points (vertical lines) report average (95\\% confidence interval) of 100 bootsrapped estimates.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figureF2", cap=cap)

# Salience of social groupy by policy issue and party ----

# load our sentence-level CAP policy topic classifications
fp <- file.path(data_path, "labeled", "uk-manifesto_sentences_lab+con_cap_labeled.tsv")
man_cap_codings <- read_tsv(fp, show_col_types = FALSE)
 
# right-join labeled sentences to manifesto info
topic_coded_man_sents <- man_dat |> 
  right_join(
    select(man_cap_codings, sentence_id, label = majortopic_recoded_label, label_prob = majortopic_recoded_score)
    , by = c("sentence_id")
  ) |> 
  select(
    manifesto_id, 
    party, partyname, date, 
    line_nr, sentence_nr, sentence_id, 
    text,
    label, label_prob
  )

set.seed(1234)
p_dat <- topic_coded_man_sents |> 
  left_join(
    all_spans |> 
      filter(label == "SG") |> 
      group_by(sentence_id) |> 
      summarise(n_sg_spans = n()) 
  ) |> 
  mutate(n_sg_spans = replace_na(n_sg_spans, 0)) |> 
  group_by(party, partyname, label) |> 
  summarise(
    n_sentences = n()
    # bootstrap salience estimate
    , bs = list(replicate(100, mean(sample(n_sg_spans > 1, replace = TRUE))))
  ) |> 
  mutate(
    q10 = map_dbl(bs, quantile, 0.1)
    , mean = map_dbl(bs, mean)
    , q90 = map_dbl(bs, quantile, 0.9)
  ) |> 
  ungroup() |> 
  filter(label != "other")

# NOTE: Figure 5
p <- p_dat |>
  group_by(label) |>
  ggplot(
    aes(
      y = reorder(label, mean)
      , x = mean
      , xmin = q10
      , xmax = q90
      , color = partyname
    )
  ) + 
  geom_point(position = position_dodge(.5), shape = 21, fill = NA, show.legend = FALSE) +
  geom_linerange(position = position_dodge(.5), show.legend = FALSE) + 
  geom_point(position = position_dodge(.5), size = .5) + 
  geom_text(
    data = p_dat |> 
      group_by(label) |> 
      filter(q90 == max(q90))
    , aes(y = reorder(label, q90), x = q90, label = label)
    , color = "black"
    , nudge_x = .01
    , size = 9/.pt
    , hjust = 0
    , vjust = .4
  ) + 
  scale_y_discrete(breaks = NULL) +
  scale_color_manual(
    breaks = names(parties_colors_map[c(1, 4)])
    , values = parties_colors_map[c(1, 4)]
  ) +
  xlim(0, .8) +
  guides(
    color = guide_legend(title = NULL, override.aes = list(shape = 15, size = 10/.pt))
  ) +
  labs(
    y = NULL
    , x = "Share of sentences with at least one social group mention"
  ) + 
  theme(legend.position = "top")

p

note_cap <- paste(
  "Sentences CAP-coded using multiclass classifier trained on human-labeled manifestos of same cases \\citep{jennings_agenda_2011}",
  "Infrequent CAP policy topics grouped into the ``other'' category.",
  "Topic ``Immigration'' recoded to topic ``Civil Rights, Minority Issues, Immigration and Civil Liberties''."
)

pn <- "uk_manifestos_sg_salience_by_party_and_cap_topic"
cap <- paste(
  sprintf("Salience of social group mentions in Labour and Conservative party manifestos (%s) by \\emph{Comparative Agendas Project} (CAP) policy topic.", paste(range(topic_coded_man_sents$date), collapse = "-")),
  "\\emph{Note:}", note_cap,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figure5", cap = cap, h = 5, w = 5)


# Social group focus analysis by party (fightin' words analysis) ----

sg_spans <- all_spans |> 
  filter(label == "SG") |> 
  filter(substr(as.character(date), 1, 4) >= 2015) |> 
  select(sentence_id, text, party) |> 
  mutate(
    partyname = id2party[as.character(party)]
  ) |> 
  mutate(
    doc_id = row_number()
    , text = gsub("��", "'", text)
  )
  
p_dat <- sg_spans |>
  mutate(partyname = parties_names_map[partyname]) |>
  corpus(docid_field = "doc_id", text_field = "text") |>
  tokenize.corpus(
    lang = "en"
    , stopwords = stopwords("en")
    , skipgrams = 1:3
    , stem = FALSE
    , .verbose = FALSE
  ) |> 
  dfm() |>
  textstat_fighting_words(group.var = "partyname", .comparison = "pairwise", .pairs = "permutations")

pairs <- list(
  c("Labour", "Conservatives"), 
  c("Greens", "UKIP"),
  c("SNP", "Labour")
)

names(pairs) <- map_chr(pairs, paste, collapse = "-")

attrs <- attributes(p_dat)
p_dat <- p_dat[names(pairs)]
attrs$names <- sub("-", " vs. ", names(pairs))
attributes(p_dat) <- attrs

# NOTE: Figure 6
set.seed(1234)
p <- textplot_fighting_words.pairwise.zscores(p_dat, k = 20, facets = FALSE, .ncol = 7)

p$layers[[3]]$aes_params$size <- 2.5
p$layers[[3]]$aes_params$segment.size <- 0.25

color_map <- parties_short_colors_map[names(parties_short_colors_map) %in% unlist(pairs)]
p <- p + 
  scale_color_manual(breaks = names(color_map), values = color_map) +
  guides(alpha = "none") + 
  labs(color = NULL)

note_fighting_words <- paste(
  "$z$-scores indicate words ``distinctiveness'' and have been obtained by applying the ``fightin' words'' method",
  "proposed by \\citet{monroe_fightin_2008} to the social group mentions retrieved by our classifier."
)

pn <- "uk_manifestos_sg_fighting_words"
cap <- paste(
  "Comparisons of different pairs of parties in terms of the words and phrases that", 
  "distinguish the social groups the mention in their manifestos for the elections 2015, 2017, and 2019.",
  "\\emph{Note:}", note_fighting_words,
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figure6", cap=cap, w = 6.5, h = 4.5)

