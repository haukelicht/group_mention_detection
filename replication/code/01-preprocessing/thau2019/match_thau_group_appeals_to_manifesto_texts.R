# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Match Thau (2018) group appeal annotations in sentence-level manifesto texts
#' @desc   Mads Thau has kindly shared his raw group appeals data.
#'         These data represent verbatim group references in the election 
#'          manifestos of the UK Labour and Conservative parties between 1664 and 2015.
#'         In this script, I detect sentences that are matching the group references
#'          extracted by Thau.
#'           
#' @author Hauke Licht
#' @date   2021-11-16
#' @update 2023-04-18
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(future)
plan(multisession, workers = 7L)
library(furrr)
library(ggplot2)
library(jsonlite)
library(stringr)

data_path <- file.path("replication", "data")

# prepare Thau (2018) annotatations ----

# read
fp <- file.path(data_path, "exdata", "thau2019", "thau2019_original_group_appeals_annotations.csv")
raw_data <- read_csv(fp, col_types = "iffiiccc")

raw_data <- raw_data %>% 
  mutate(
    party_id = ifelse(sourname == 1, 51620L, 51320L)
    , party_name = ifelse(sourname == 1, "Conservative Party", "Labour Party")
  ) 

table(raw_data$objdim, useNA = "ifany")

# format document names
doc_mapping <- raw_data %>% 
  select(party_name, year1, year2, month) %>% 
  unique() %>% 
  mutate(
    tmp_ = ifelse(
      grepl("(\\d{4})\\.(\\d)", year1)
      , sprintf("%s-%02d", substr(year2, 1, 4), month)
      , substr(year2, 1, 4)
    )
    , doc_name = paste0(
      ifelse(party_name == "Conservative Party", "conservatives", "labour")
      , "-"
      , tmp_
      , ".md"
    )
    , tmp_ = NULL
  )

length(table(doc_mapping$doc_name))

# add information 
raw_data <- left_join(raw_data, doc_mapping)

# check if all are utf-8 readable
warns <- tryCatch(grepl("", raw_data$objid, fixed = TRUE), warning = function(wrn) wrn)
warns$message

# clean the one instance with an issue
raw_data$objid[2339]
raw_data$objid[2339] <- sub("\xe9s", "é", raw_data$objid[2339])

#' @note
#' Mads Thau writes in his [Coding Instructions](https://politica.dk/fileadmin/politica/Dokumenter/Afhandlinger/mads_thau.pdf) (p. 83) for the `objid` dimension:
#'    
#'    Write name exactly as it figures. 
#'    Include the entire description of the object. 
#'    If the text says “Many families who live on council estates and in new towns 
#'     would like to buy their own homes but either cannot afford to or are prevented 
#'     by the Labour government” (example from table 9) the full “Many families who 
#'     live on council estates and in new towns” denotes the object identity and 
#'     should be recorded
#'    
#'    Further, in the case of a coupled object that needs to be separated into 
#'     multiple appeals, preserve the full text in each appeal but put brackets 
#'     around the part that does not make up that observation.
#'    In our example write “Many families (who live on council estates and in 
#'     new towns)” when coding the family appeal, and “(Many families) who live
#'     on council estates (and in new towns)” when coding the council tenant 
#'     appeal, and 
#'     “(Many families who live  84 on council estates and in) new towns” 
#'     when coding the one about new town tenants.
#'    
#'    Lastly, if coders need to add information to the object to aid subsequent 
#'     reading insert a bracket after the object and start with “i.e.”. 
#'    For instance, for an object simply reading “those who are” coders should 
#'     add information to aid interpretation. It could look like this: 
#'     “those who are (i.e. unemployed)”.
#'     
#' This implies that to identify the passages that Thau's coders have identified
#'  as objects of group appeals (i.e., group mentions) in the original manifesto 
#'  texts, we need to 
#'  
#'  1. remove the parentheses
#'  2. extract the content inside square brackets (this can be used to disambiguate between multiple match candidates because the context words should occur in the same sentence)
#'  3. try to match these reconstructed verbatim mentions to the manifesto text 
#'
#' Sounds easy.
#' However, the thing is that Thau's coders haven't consistently applied these rules.
#' This can be seen, first, by the fact that there are very few instances that exhibit the "[i.e., " patterns
raw_data %>% filter(grepl("\\[i\\.?e\\.?", objid))
#' Coders just often omitted the "i.e.," after the opening square bracket
raw_data %>% filter(grepl("[", objid, fixed = TRUE))
#' And others have used the i.e., but inside _parentheses_
raw_data %>% filter(grepl("(i.e.", objid, fixed = TRUE))
#' Fortunately, at least they seem to have followed the instruction for using parentheses to segment nested/inersectional spans
raw_data %>% filter(grepl("(", objid, fixed = TRUE))

#' To find obervations I've made:
#' 1. another pattern indicates passages where they've omitted words in between span segments
raw_data %>% filter(grepl("...", objid, fixed = TRUE))
#' 2. theres is one instance where the coder indicated a grammatical error using "[sic]" (need to remove)
raw_data %>% filter(grepl("[sic]", objid, fixed = TRUE))

# I clean the data step by step:
cleaned_data <- raw_data %>% 
  select(doc_name, party_id, party_name, id, objdim, objid) %>% 
  # prepare:
  mutate(
    # retain the original records
    objid_original = objid,
    #remove [sic] from the one instance
    objid = sub("[sic] ", "", objid, fixed = TRUE)
  ) %>% 
  # (i) extract content from fields 'objid' that have the pattern "[i.e."
  mutate(
    context_words = str_extract_all(objid, "(?<=\\[i\\.?e\\.?)[^\\]]+(?=\\])"),
    flag = lengths(context_words) > 0, # keep track of which already cleaned
    objid = objid %>% 
      str_remove_all("\\[i\\.?e\\.?,?\\s+[^\\]]+\\]") %>% 
      str_trim()
  ) %>% 
  # (ii) handle the case where they used parenthese instead of square brackets
  mutate(
    context_words = ifelse(flag, context_words, str_extract_all(objid, "(?<=\\(i\\.?e\\.?)[^\\)]+(?=\\))")),
    flag = lengths(context_words) > 0,
    objid = objid %>% 
      str_remove_all("\\(i\\.?e\\.?,?\\s+[^\\)]+\\)") %>% 
      str_trim()
  ) %>% # select(doc_name, objid_original, objid, context_words) %>%  sample_n(10) %>% unnest_longer(context_words)
  # (iii) now treat cases where the "i.e." was omitted inside the square brackets
  mutate(
    context_words = ifelse(flag, context_words, str_extract_all(objid, "(?<=\\[)[^\\]]+(?=\\))")),
    flag = lengths(context_words) > 0,
    objid = objid %>% 
      str_remove_all("\\[[^\\]]+\\]") %>% 
      str_trim()
  ) %>% 
  # (iv) remove remaining parentheses to get the spans that contain the mentions
  mutate(
    flag = ifelse(flag, flag, str_count(objid, fixed("(")) > 0),
    objid = objid %>% 
      str_remove_all("[()]") %>% 
      str_replace_all("\\s{2,}", " ") %>% 
      str_trim()
  ) %>% # filter(flag) %>% sample_n(10)
  # (v)
  # add special token where extra words occur between start and end of span
  mutate(
    objid = str_replace_all(objid, fixed("..."), "<WORDS>")
  ) %>% 
  # finally: clean up
  mutate(
    context_words = map(context_words, str_remove_all, pattern = "^\\.?,?\\s")
  ) %>% 
  rename(
    we_cleaned = flag
    , verbatim_mention = objid
    , objid = objid_original
  )

# write_rds(cleaned_data, "data/thau_2018/cleaned_spans.rds")

# match to manifesto texts ----

manifesto_texts <- read_rds("data/manifestos/all_uk_manifesto_sentences.rds")
fp <- file.path(data_path, "corpora", "uk-manifesto_sentences.tsv")
manifesto_texts <- read_tsv(fp, show_col_types = FALSE)

these_docs <- cleaned_data %>% 
  mutate(date = sub(".+(\\d{4})-?(\\d{2})?\\.md$", "\\1\\2", doc_name)) %>% 
  distinct(date, party_id, doc_name_thau = doc_name)

# subset to relevant  manifestos 
manifesto_texts$date <- as.character(manifesto_texts$date)
manifesto_texts <- right_join(manifesto_texts, these_docs, by = c("party" = "party_id", "date"))

# verify 
View(distinct(manifesto_texts, date, party, doc_name_thau))

match_thaus_extracted_spans_to_manifesto_texts <- function(x, ref, to.lower = TRUE, to.latin.chars = TRUE, .successively = FALSE) {
  
  out <- x
  if (to.latin.chars) {
    out$verbatim_mention <- stringi::stri_trans_general(out$verbatim_mention, id = "Any-Latn") 
    out$context_words <- map(out$context_words, stringi::stri_trans_general, id = "Any-Latn")
    ref$text <- stringi::stri_trans_general(ref$text, id = "Any-Latn") 
  }
  if (to.lower) {
    out$verbatim_mention <- tolower(out$verbatim_mention) 
    out$context_words <- map(out$context_words, tolower)
    ref$text <- tolower(ref$text) 
  }
  
  i <- 1L
  n_ <- nrow(ref)
  out <- split(out, x$id)
  
  for (j in seq_along(out)) {
    if (i > n_) break
    
    (pat <- str_split(out[[j]]$verbatim_mention, "<WORDS>")[[1]])
    (cw <- out[[j]]$context_words[[1]])
    
    idxs <- map_lgl(
      ref$text[i:n_], 
      function(text) {
        flag <- all(map_lgl(pat, ~grepl(., text, fixed = TRUE)))
        if (!flag)
          return(FALSE)
        if (length(cw) == 0)
          return(TRUE)
        all(map_lgl(cw, ~grepl(., text, fixed = TRUE)))
      }
    )
    if (!any(idxs))
      next
    
    tmp <- list()
    for (idx in which(idxs)) { # idx <- which(idxs)[[1]]
      span <- map(pat, ~regexec(., ref$text[i:n_][idx], fixed = TRUE))
      span <- purrr::flatten(span)
      span <- data.frame(s = map_int(span, as.integer), l = map_int(span, attr, "match.length"))  
      span$e <- span$s+span$l-1
      span <- summarize(span, s = min(s), e = max(e))
      span$sentence_id <- ref$sentence_id[i:n_][idx]
      span$text <- ref$text[i:n_][idx]
      span$span <- substr(ref$text[i:n_][idx], span$s, span$e)
      span
      tmp[[length(tmp)+1L]] <- span
    }
    out[[j]]$matched <- list(do.call(rbind, tmp))
    
    # update
    if (.successively)
      i <- min(which(idxs))
  }
  
  out <- unnest(bind_rows(out), matched)
  return(out)
}


# extract matches
matched_spans <- future_map2_dfr(
  split(cleaned_data, cleaned_data$doc_name)[these_docs$doc_name_thau]
  , split(manifesto_texts, manifesto_texts$doc_name_thau)[these_docs$doc_name_thau]
  , match_thaus_extracted_spans_to_manifesto_texts
  , .successively = TRUE
  , .progress = TRUE
)

table(cleaned_data$id %in% matched_spans$id) %>% prop.table()

# quite a lot of entries in Thau's "objid" column are matched mutiple times within manifesto
tmp <- matched_spans %>% 
  group_by(id) %>% 
  summarise(n_matched = n_distinct(sentence_id)) %>% 
  ungroup()
  
ggplot(tmp, aes(x = n_matched)) + 
  geom_histogram() +
  scale_y_log10()
  
quantile(tmp$n_matched, c(.1, .25, .5, .8, .9, .95, .99))

# IMPORTANT: keep only tos that are matched at most 3 times
matched_spans <- tmp %>% 
  filter(n_matched <= 3) %>% 
  select(id) %>% 
  left_join(matched_spans)

# write to disk ----

fp <- file.path(data_path, "exdata", "thau2019", "thau2019_spans_matched_to_manifesto_texts.rds")
# write_rds(matched_spans, fp)

matched_spans <- mutate(matched_spans, context_words = map_chr(context_words, paste, collapse = "<SEP>"))
fp <- sub("rds$", "tsv", fp)
if (!file.exists(fp))
  write_tsv(matched_spans, fp)

tmp <- matched_spans %>% 
  left_join(these_docs, by = c("party_id", "doc_name" = "doc_name_thau")) %>% 
  mutate(
    label = ifelse(is.na(objdim), "none", objdim)
    # decrement start (because of zero-indexing in pyhton)
    , s = s-1
  ) %>% 
  select(party = party_id, date, doc_name, sentence_id, text, s, e, label, id) %>% 
  group_by(party, date, doc_name, sentence_id, text) %>% 
  nest(data = c(s, e, label, id)) %>% 
  as.data.frame()

jsonify <- function(x) {
  out <- list(
    "id" = x$sentence_id
    , "text" = x$text
    , "label" = x$data[[1]]
    , metadata = as.list(x[c("party", "date", "doc_name")])
  )
  out <- jsonlite::toJSON(out, dataframe = "values", auto_unbox = TRUE)
  return(out)
}

lines <- map_chr(split(tmp, 1:nrow(tmp)), jsonify)
fp <- sub("rds$", "jsonl", fp)
if (!file.exists(fp))
  write_lines(lines, fp)
