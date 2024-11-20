# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Describe the dataset
#' @author Hauke Licht
#' @date   2023-04-17
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----
library(readr)
library(dplyr)
library(tidyr)
library(purrr)


# TODO: adjust path
data_path <- file.path("replication", "data")
floats_path <- file.path("replication", "paper")
utils_path <- file.path("replication", "code", "r_utils")

source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)

# UK manifestos ----

fp <- file.path(data_path, "corpora", "uk-manifesto_sentences.tsv")
uk_mans <- read_tsv(fp)

parties <- read_tsv(file.path(data_path, "uk_parties.tsv"))

uk_mans <- uk_mans %>% 
  left_join(parties, by = c("partyname" = "party_name")) %>% 
  mutate(
    date = ifelse(substr(date, 1 , 4) >= 2015, substr(date, 1 , 4), date)
    , date = sub("(?<=^\\d{4})(?=\\d{2}$)", "-", date, perl = TRUE)
  ) 

# NOTE: Table A1
uk_mans %>% 
  left_join(parties, by = c("partyname" = "party_name", "party_name_short")) %>% 
  mutate(
    date = ifelse(date == "201505", "2015", date)
    , date = sub("(?<=^\\d{4})(?=\\d{2}$)", "-", date, perl = TRUE)
  ) %>% 
  count(party_name_short, date) %>% 
  pivot_wider(names_from = date, values_from = n) %>% 
  arrange(party_name_short) %>% 
  rename(Party = party_name_short) %>% 
  quick_kable(
    caption = "Number of sentences in UK party manifestos."
    , align = c("l", rep("r", ncol(.)-1))
    , label = "num_sentences_uk-manifestos"
  ) %>% 
  save_kable(.file.name = "tableA1")


# DE manifestos ----

fp <- file.path(data_path, "corpora", "de-manifesto_sentences.tsv")
de_mans <- read_tsv(fp)

# NOTE: Table A2
de_mans %>% 
  arrange(party_name, date) %>% 
  count(party_name, date) %>% 
  pivot_wider(names_from = date, values_from = n) %>% 
  arrange(party_name) %>% 
  rename(Party = party_name) %>% 
  select(Party, !!sort(as.character(unique(de_mans$date)))) %>% 
  quick_kable(
    caption = "Number of sentences in German party manifestos."
    , align = c("l", rep("r", ncol(.)-1))
    , label = "num_sentences_de-manifestos"
  ) %>% 
  save_kable(.file.name = "tableA2")


# UK Commons ----

fp <- file.path(data_path, "corpora", "uk-commons_2013-2019_sentences.tsv")
dat <- read_tsv(fp)

tmp <- dat %>% 
  group_by(party, year = lubridate::year(date)) %>% 
  summarise(
    n_speakers = n_distinct(speaker)
    , n_speeches = n_distinct(speech_nr_parlspeech, speech_nr)
    , n_sentenes = n_distinct(speech_nr_parlspeech, speech_nr, sent_nr)
  ) %>% 
  ungroup()

# NOTE: Table A3
tmp %>% 
  select(Party = party, year, value = n_sentenes) %>% 
  pivot_wider(names_from = year) %>% 
  quick_kable(
    caption = "Number of sentences in UK House of Commons data."
    , align = c("l", rep("r", ncol(.)-1))
    , label = "num_sentences_uk-parlspeech"
  ) %>% 
  save_kable(.file.name = "tableA3")
