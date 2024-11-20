# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Describe mentions recorded in Thau (2019) data
#' @author Hauke Licht
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load packages
library(readr)
library(dplyr)

data_path <- file.path("replication", "data")
utils_path <- file.path("replication", "code", "r_utils")
floats_path <- file.path("replication", "paper")

# table setup
source(file.path(utils_path, "table_setup.R"))
tabs_dir <- file.path(floats_path, "tables")
save_kable <- partial(save_kable, dir = tabs_dir, overwrite = TRUE)

parties <- read_tsv(file.path(data_path, "uk_parties.tsv"))
party2id <- with(parties, setNames(party, party_name))
id2party <- with(parties, setNames(party_name, party))

# describe Thau data ----

# load mentions recorded in Thau (2019) data
fp <- file.path(data_path, "exdata", "thau2019", "thau2019_appeals_appeal.csv")
thau_dat <- read_csv(fp, show_col_types = FALSE)

thau_dat <- thau_dat |> 
  select(year1, month, claimpar, objtype, objid, objdim) |> 
  mutate(
    party = party2id[claimpar],
    partyname = id2party[as.character(party)],
    date = ifelse(year1 != round(year1, 0),
                  as.integer(sprintf("%d%02d", as.integer(year1), month)),
                  as.integer(year1)
    ),
    year = as.integer(year1)
  )

# NOTE: Table F1
thau_dat |> 
  group_by(objtype) |> 
  summarise(
    n_mentions = n(),
    n_unique_mentions = n_distinct(objid)
  ) |> 
  arrange(objtype == "Other", objtype) |> 
  quick_kable(
    caption = paste(
      "Number of (unique) mentions of different group types identified in",
      sprintf("Labour and Conservative party manifestos (%d-%d)", min(thau_dat$year), max(thau_dat$year)),
      "by \\citet{thau_how_2019}.",
      collapse = " "
    )
    , col.names = c("Group type", "$n$ mentions", "$n$ unique mentions")
    , label = "thau2019_group_type_counts"
    , align = c("l", "r", "r")
  ) |> 
  save_kable(.file.name = "tableF1",  position = "!h")


# subset to social group mentions
thau_dat <- filter(thau_dat, objtype == "Social group") 

# NOTE: Table F2
thau_dat |> 
  group_by(objdim) |>
  summarise(
    n_mentions = n(),
    n_unique_mentions = n_distinct(objid)
  ) |> 
  arrange(objdim == "Other", objdim) |> 
  quick_kable(
    caption = paste(
      "Number of (unique) social group mentions identified in",
      sprintf("Labour and Conservative party manifestos (%d-%d)", min(thau_dat$year), max(thau_dat$year)),
      "by \\citet{thau_how_2019}",
      "across different social group categories.",
      collapse = " "
    )
    , col.names = c("Social group category", "$n$ mentions", "$n$ unique mentions")
    , label = "thau2019_social_group_category_counts"
    , align = c("l", "r", "r")
  ) |> 
  save_kable(.file.name = "tableF2", position = "!h")
