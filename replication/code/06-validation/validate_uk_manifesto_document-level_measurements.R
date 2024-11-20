# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Analyze spans extracted from UK parties' manifestos
#' @author Hauke Licht
#' @date   2022-11-18
#' @update 2023-04-14, 2023-11-24, 2024-08-06, 2024-09-07, 2024-09-14
#' @note:  Spans extracted with Roberta (base) token classifier trained on
#'          labels aggregated from RA's annotations from all UK manifestos
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(purrr)
library(ggplot2)
library(quanteda)

# TODO: update paths
data_path <- file.path("replication", "data")
utils_path <- file.path("replication", "code", "r_utils")
floats_path <- file.path("replication", "paper")

# figure setup
source(file.path(utils_path, "plot_setup.R"))
figs_dir <- file.path(floats_path, "figures")
save_plot <- partial(save_plot, fig.path=figs_dir)

parties_colors <- read_tsv(file.path(data_path, "uk_parties.tsv"))
parties_names_map <-  with(parties_colors, set_names(party_name_short, party_name))
parties_short_names_map <- with(parties_colors, set_names(party_name_short, party_name))
parties_colors_map <- with(parties_colors, set_names(color_dark, party_name))
parties_short_colors_map <- with(parties_colors, set_names(color_dark, party_name_short))

# load data ----

man_dat <- read_tsv(file.path(data_path, "corpora", "uk-manifesto_sentences.tsv"))
parties <- distinct(man_dat, party, partyname)
party2id <- with(parties, setNames(party, partyname))
id2party <- with(parties, setNames(partyname, party))

all_spans <- read_tsv(file.path(data_path, "labeled", "uk-manifesto_sentences_predicted_spans.tsv"))

# load  mentions recorded in Thau (2019) data ----

fp <- file.path(data_path, "exdata", "thau2019", "thau2019_appeals_appeal.csv")
thau_dat <- read_csv(fp, show_col_types = FALSE)

thau_dat <- thau_dat %>% 
  select(year1, month, claimpar, objtype, objid, objdim) %>% 
  mutate(
    party = party2id[claimpar],
    partyname = id2party[as.character(party)],
    date = ifelse(year1 != round(year1, 0),
                  as.integer(sprintf("%d%02d", as.integer(year1), month)),
                  as.integer(year1)
    ),
    year = as.integer(year1)
  )

# validate manifesto-level measurements ----

## ours against Thau's ----

# compute number of mentions predicted by our classifier at manifesto level
manifestos_n_sg_spans <- all_spans %>% 
  filter(label == "SG") %>% 
  count(party, date)

corr_all <- thau_dat %>% 
  count(party, year, date, name = "thau") %>% 
  left_join(manifestos_n_sg_spans,  by = c("party", "date")) %>% 
  # arrange(party, year) %>% 
  with(cor.test(thau, n))

# NOTE: Figure 3
p <- thau_dat %>% 
  count(party, year, date, name = "thau") %>% 
  left_join(manifestos_n_sg_spans,  by = c("party", "date")) %>% 
  mutate(
    partyname = ifelse(party == 51620, "Conservatives", "Labour")
  ) %>% 
  ggplot(aes(x = thau, y = n, color = partyname)) +
    geom_abline(slope = 1, lwd = .25) + 
    geom_point() +
    scale_color_manual(
      values = parties_short_colors_map[c("Conservatives", "Labour")]
      , labels = c("Conservative Party", "Labour Party")
    ) +
    xlim(0, 1100) + 
    ylim(0, 1100) +
    annotate(
      geom = "text"
      , x = 200, y = 1050
      , label = with(corr_all, sprintf("r = %.02f [%.02f, %.02f]", estimate, conf.int[1], conf.int[2]))
      , size = 9/.pt
    ) +
    guides(
      color = guide_legend(title = NULL, override.aes = list(shape = 15, size = 10/.pt))
    ) + 
    labs(
      x = "No. mentions recorded in Thau (2019) data"
      , y = "No. mentions extracted by classifier"
      , color = NULL
    ) + 
    theme(legend.position = "top")

pn <- "uk_manifestos_sg_counts_cross_validation"
cap <- paste(
  "Cross validation of RoBERTa group mention detection classifier's predictions against data collected by \\citet{thau_how_2019}.",
  "Figure compares the numbers of social group mentions identified in a manifesto by \\citet[see x-axis]{thau_how_2019} and our classifier (y-axis)",
  sprintf("in Labour and Conservative party manifestos (%d-%d).", min(thau_dat$year), max(thau_dat$year)),
  "Colors indicate parties.",
  "Correlation coefficient (with 95\\% confidence interval) shown in top-left of plot panel.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figure3", cap = cap, h = 4, w = 3.8)

## Dolinsk-Huber-Horne (DHH) dictionary-based against Thau's ----

#' note: because the dictionary indexes words and not word spans (mentions),
#'     it doesn't make sense to correlate the number of extracted mentions.
#'   For example, the dictionary might detect several words that belong to one 
#'    mention.
#'   Instead, we correlate the share of sentences that contain 1+ social group 
#'    mentions/keyword hits.

fp <- file.path(data_path, "exdata", "dhh_dictionary", "keywords.csv")
keywords_wider <- read_csv(fp)

keywords <- map(as.list(as.data.frame(keywords_wider)), compose(as.vector, na.omit))

# number of categories
length(keywords)

# number of entries
sum(lengths(keywords))

# number of unique terms
keywords %>%
  unlist() %>%
  strsplit("\\s+") %>%
  unlist() %>%
  trimws() %>%
  n_distinct()

# distribution of terms per entry
keywords %>%
  unlist() %>%
  strsplit("\\s+") %>%
  lengths() %>%
  tibble(n = .) %>%
  count(n, name = "n_times") %>%
  mutate(
    prop = n_times/sum(n_times),
    cum_prop = cumsum(prop)
  )
# - 47.6% of entries have two or more terms
# - 15.9% of entries have three or more terms

### compare number of terms agianst THau (2019) -----

dhh <- keywords_wider %>% 
  pivot_longer(cols=everything(), values_to = "words", names_to = "dimension") %>% 
  filter(!is.na(words)) %>% 
  mutate(
    objdim = case_when(
      dimension %in% c("var1", "var10", "var11", "var2", "var3", "var37", "var38", "var4", "var5", "var6", "var7", "var8", "var9")  ~ "Economic class" ,
      dimension %in% c("var12", "var13", "var14", "var19", "var20", "var21", "var22", "var41", "var42", "var43") ~ "Other" ,
      dimension %in% c("var15", "var18", "var30", "var31")  ~ "Nationality",
      dimension %in% c("var16", "var39", "var40") ~ "Geography",
      dimension %in% c("var17") ~ "Ethnicity/race",
      dimension %in% c("var23", "var24", "var25")  ~ "Age/generation",
      dimension %in% c("var26", "var27", "var28")  ~ "Gender",
      dimension %in% c("var29") ~ "Health",
      dimension %in% c("var32", "var33", "var34", "var35", "var36") ~ "Religion"
    )
  )

sum_dhh <- dhh %>% 
  group_by(objdim, dat = "DHH") %>%
  summarize(count = n_distinct(words))


sum_thau <- thau_dat %>%
  filter(objtype == "Social group") %>%
  group_by(objdim, dat = "Thau") %>%
  summarize(count = n_distinct(objid))

# NOTE: Figure 1
p <- bind_rows(sum_thau, sum_dhh) |> 
  ggplot(aes(x = reorder(objdim, desc(objdim)), y = count, fill = dat))+
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_grey() +
  ylim(0, 1500) +
  geom_text(
    aes(label = count),
    position = position_dodge(width = 0.9),
    hjust = -0.25,
    size = 7/.pt
  ) +
  labs(
    x = "Social group category", 
    y = "Number of unique n-grams", 
    fill = "Data source"
  ) + 
  coord_flip()
p
pn <- "thau_n_grams_by_category"
cap <- paste(
  "Number of unique n-grams",
  "in human-annotated data collected by \\citet{thau_how_2019}",
  "and in the Dolinsky-Huber-Horne (DHH) dictionary compiled by \\citet{dolinsky_parties_2023}",
  "by social group category.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figure1", cap = cap, w = 4.5, h = 3.5)


# convert to quanteda dictionary
dhh_dictionary <- quanteda::dictionary(keywords, tolower = FALSE)

man_sentences_with_dhh_measures <- man_dat %>% 
  select(sentence_id, text) %>%
  corpus(docid_field = "sentence_id", text_field = "text") %>% 
    tokens(
      what = "word",
      remove_numbers = FALSE,
      remove_separators = FALSE,
      split_hyphens = FALSE
    ) %>% 
    dfm() %>% 
    dfm_lookup(dictionary = dhh_dictionary, valuetype = "glob") %>% 
    # summarize matches across dictionary categories
    rowSums() %>%
    # convert to binary indicator (1+ matches or none)
    sign() %>%
    tibble::enframe(name = "sentence_id", value = "has_sg_mention") %>% 
    left_join(select(man_dat, sentence_id, manifesto_id, date, party, partyname))

man_dhh_n_sg_spans <- count(man_sentences_with_dhh_measures, party, date)
  
man_dhh_n_sg_spans$date <- as.integer(man_dhh_n_sg_spans$date)

corr_all_dhh <- thau_dat %>% 
  count(party, year, date, name = "thau") %>% 
  left_join(man_dhh_n_sg_spans,  by = c("party", "date")) %>% 
  # arrange(party, year) %>% 
  with(cor.test(thau, n))

corr_all_dhh

# NOTE: Figure G1
p <- thau_dat %>% 
  count(party, year, date, name = "thau") %>% 
  left_join(man_dhh_n_sg_spans,  by = c("party", "date")) %>% 
  mutate(
    partyname = ifelse(party == 51620, "Conservatives", "Labour")
  ) %>% 
  ggplot(aes(x = thau, y = n, color = partyname)) +
  geom_abline(slope = 1, lwd = .25) + 
  geom_point() +
  scale_color_manual(
    values = parties_short_colors_map[c("Conservatives", "Labour")]
    , labels = c("Conservative Party", "Labour Party")
  ) +
  xlim(0, 1700) + 
  ylim(0, 1700) +
  annotate(
    geom = "text"
    , x = 400, y = 1700
    , label = with(corr_all_dhh, sprintf("r = %.02f [%.02f, %.02f]", estimate, conf.int[1], conf.int[2]))
    , size = 9/.pt
  ) +
  guides(
    color = guide_legend(title = NULL, override.aes = list(shape = 15, size = 10/.pt))
  ) + 
  labs(
    x = "No. mentions recorded in Thau (2019) data"
    , y = "No. mentions extracted by Dolinsky-Huber-Horne dictionary"
    , color = NULL
  ) + 
  theme(legend.position = "top")

pn <- "uk_manifestos_dhh_sg_sentence_counts_cross_validation"
cap <- paste(
  "Cross validation of sentence classifications generated with Dolinsky-Huber-Horne group keyword dictionary against data collected by \\citet{thau_how_2019}.",
  "Figure compares the numbers of social group mentions identified in a manifesto by \\citet[see x-axis]{thau_how_2019} and",
  "the number of sentences containing at least one group keyword recorded in the Dolinsky-Huber-Horne dictionary (y-axis).",
  sprintf("in Labour and Conservative party manifestos (%d-%d).", min(thau_dat$year), max(thau_dat$year)),
  "Colors indicate parties.",
  "Correlation coefficient (with 95\\% confidence interval) shown in top-left of plot panel.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figureG1", cap = cap, h = 4, w = 3.8)


## ours against DHH dictionary-based measurements ----

man_sentences_n_sg_mentions <- all_spans %>% 
  filter(label == 'SG') %>% 
  count(party, date, sentence_id, name = "has_sg_mention") %>% 
  mutate(date = as.integer(date)) %>%
  right_join(
    transmute(man_dat, party, partyname, date = as.integer(date), sentence_id)
  ) %>% 
  arrange(party, date, sentence_id) %>% 
  mutate(has_sg_mention = as.integer(!is.na(has_sg_mention)))

tmp <- man_sentences_with_dhh_measures %>%
  select(sentence_id, has_sg_mention_dhh = has_sg_mention) %>%
  inner_join(man_sentences_n_sg_mentions) %>% 
  left_join(distinct(thau_dat, date, party, in_thau = TRUE)) %>% 
  mutate(in_thau = replace_na(in_thau, FALSE))

pdat <- tmp %>% 
  group_by(party, partyname, date) %>%
  summarise(
    in_thau = all(in_thau),
    sg_salience_dhh = mean(has_sg_mention_dhh),
    sg_salience_ours = mean(has_sg_mention),
    NULL
  )

(corr_man_level <- with(pdat, cor.test(sg_salience_dhh, sg_salience_ours)))
(corr_man_level <- with(pdat[pdat$in_thau, ], cor.test(sg_salience_dhh, sg_salience_ours)))

# finding: measures correlate strongly at document level of aggregation

# NOTE: Figure G2
p <- pdat %>% 
  # subset to cases in Thau data for comparability
  filter(in_thau) %>%
  ggplot(aes(x = sg_salience_ours, y = sg_salience_dhh, color = parties_names_map[partyname])) +
  geom_abline(slope = 1, lwd = .25) + 
  geom_point() +
  scale_color_manual(
    values = parties_short_colors_map
    , labels = names(parties_short_colors_map)
  ) +
  xlim(0, .5) +
  ylim(0, .5) +
  annotate(
    geom = "text"
    , x = 0.1, y = .5
    , label = with(corr_man_level, sprintf("r = %.02f [%.02f, %.02f]", estimate, conf.int[1], conf.int[2]))
    , size = 9/.pt
  ) +
  guides(
    color = guide_legend(ncol = 4, label.hjust = 0, override.aes = list(shape = 15, size = 10/.pt))
  ) + 
  labs(
    x = "Share of sentences with min. one social group mention\ndetected by our classifier"
    , y = "Share of sentences with min. one social group mention\ndetected with Dolinsky-Huber-Horne dictionary"
    , color = NULL
  ) + 
  theme(legend.position = "top")

p
pn <- "uk_manifestos_sg_salience_ours_vs_dhh"
cap <- paste(
  "Cross validation of social group salience measures generated with our classifier against Dolinsky-Huber-Horne dictionary.",
  "Figure compares the share of sentences per manifesto that contain at least one social group mention according to our classifier to",
  "the share of sentences per manifesto that contain at least one group keyword recorded in the Dolinsky-Huber-Horne dictionary (y-axis).",
  "Colors indicate parties.",
  "Cases subset to those in Thau (2019) data for the sake of comparability.",
  "Correlation coefficient (with 95\\% confidence interval) shown in top-left of plot panel.",
  sprintf("\\label{fig:%s}", pn),
  collapse = " "
)
save_plot(p, fn = "figureG2", cap = cap, h = 4, w = 3.8)


