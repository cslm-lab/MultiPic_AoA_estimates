# DEFINE CUSTOM FUNCTIONS

library(tibble)
library(readr)
library(readxl)
library(readODS)
library(dplyr)
library(magrittr)
library(ggplot2)
#library(RColorBrewer)
#library(cowplot)
library(stringr)
################################################################################
load_raw_data <- function(){
  # Loads raw item-based data and adds another column translating the 
  # continuous AoA estimate into a Likert rating.
  aoa_info_original <- read_csv("../data/raw/derivatives/item_based_data_add_info.csv")
  # add column: estimates in 7-point Likert scale
  aoa_info <- aoa_info_original %>% 
    rowwise() %>% 
    mutate(estimateLikert = rating_to_likert(estimate), .after = estimate)
  return(aoa_info)
}

remove_umlauts <- function(input_string){
  # Removes umlauts and ß and lowercases strings.
  #  Input:
  #      input_string: A string.
  #  Output:
  #      new_string: Same string in lowercase and without umlauts.
  new_string <- str_to_lower(input_string)
  new_string <- str_replace_all(new_string, "ß", "ss")
  new_string <- str_replace_all(new_string, "ä", "ae")
  new_string <- str_replace_all(new_string, "ö", "oe")
  new_string <- str_replace_all(new_string, "ü", "ue")
}

rating_to_likert <- function(input_numerical){
  # Turns an AoA estimate into a Likert rating.
  # Likert scale after Schröder et al.:  
  # 1 = 0–2 years, 2 = 3–4 years, 3 = 5–6 years, 4 = 7–8 years, 5 = 9–10 years, 6 = 11–12 years, 7 = 13+ years
  # Input: 
  #   input_numerical: a raw AoA estimate.
  # Output:
  #   likert_estimate: raw estimate turned into its corresponding Likert bin.
  if (is.na(input_numerical)) {
    return(NA)
  } else if (input_numerical <= 2) {
    likert_estimate <- 1
  } else if (input_numerical > 2 & input_numerical <= 4) {
    likert_estimate <- 2
  } else if (input_numerical > 4 & input_numerical <= 6) {
    likert_estimate <- 3
  } else if (input_numerical > 6 & input_numerical <= 8) {
    likert_estimate <- 4
  } else if (input_numerical > 8 & input_numerical <= 10) {
    likert_estimate <- 5
  } else if (input_numerical > 10 & input_numerical <= 12) {
    likert_estimate <- 6
  } else {
    likert_estimate <- 7
  }
  return(likert_estimate)
}

get_estimates_overview <- function(raw_estimates_table, all=TRUE){
  # Creates a new overview table displaying the mean, sd, min and max values of 
  # both the continuous estimates and the transformed Likert estimates for
  # the given data.
  # Input:
  #   raw_estimates_table: aoa_info or a subset thereof
  # Output:
  #   aoa_estimates: an overview table over group estimates
  
  # get unique MultiPic item numbers
  multipic_items <- sort(unique(raw_estimates_table$item_number))
  # fill in item names and AoA group estimates
  if (all==FALSE){
    # create tibble for AoA estimate collection
    aoa_estimates <- tibble("item_number" = multipic_items, "estimate_mean" = 0, "estimate_sd" = 0)
    for (item_nb in multipic_items) {
      # get all estimates for current item number (without NA)
      item_data <- filter(raw_estimates_table, item_number == item_nb, !is.na(estimate))
      # add mean estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimate_mean <- mean(item_data$estimate)
      # add standard deviation of estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimate_sd <- sd(item_data$estimate)
    }
  } else {
    # create tibble for AoA estimate collection
    aoa_estimates <- tibble("item_number" = multipic_items, "estimate_mean" = 0, "estimate_sd" = 0, "min" = 0, "max" = 0, "estimateLikert_mean" = 0, "estimateLikert_sd" = 0, "minLikert" = 0, "maxLikert" = 0) 
    for (item_nb in multipic_items) {
      # get all estimates for current item number (without NA)
      item_data <- filter(raw_estimates_table, item_number == item_nb, !is.na(estimate))
      # add mean estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimate_mean <- mean(item_data$estimate)
      # add standard deviation of estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimate_sd <- sd(item_data$estimate)
      # add min item estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$min <- min(item_data$estimate)
      # add max item estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$max <- max(item_data$estimate)
      # add Likert mean estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimateLikert_mean <- mean(item_data$estimateLikert)
      # add Likert standard deviation of estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$estimateLikert_sd <- sd(item_data$estimateLikert)
      # add Likert min item estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$minLikert <- min(item_data$estimateLikert)
      # add Likert max item estimate to overview
      aoa_estimates[aoa_estimates$item_number == item_nb,]$maxLikert <- max(item_data$estimateLikert)
    }
  }
  return(aoa_estimates)
}

read_birchenough <- function(select_cols=TRUE){
  birchenough_norms <- read_csv('../../external_resources/norms/Birchenough_2017.csv', locale=locale(encoding="latin1"))
  # filter for useful columns
  col_rename <- c("B: AoA mean"="AoAestimate", "B: AoA SD"="SD", "B: min"="min", "B: max"="max", "B: AoALikert mean"="AoALikert", "B: AoALikert SD"="SDLikert", "B: minLikert"="minLikert", "B: maxLikert"="maxLikert")
  birchenough_norms <- birchenough_norms %>% 
    mutate(NAME1 = remove_umlauts(Word)) %>% # lowercase words + remove umlauts
    rename(all_of(col_rename))
  if(select_cols==TRUE){
    birchenough_norms <- birchenough_norms %>%
      select(NAME1, "B: AoA mean", "B: AoA SD", "B: min", "B: max", "B: AoALikert mean", "B: AoALikert SD", "B: minLikert", "B: maxLikert") 
  }
  return(birchenough_norms)
}

read_schröder <- function(select_cols=TRUE){
  # Reads in German norms from Schröder et al. (2011).
  # Lowercases + removes umlauts of target word column to make it comparable to MultiPic,
  # if select_cols=TRUE: selects only the target name + AoA rating columns .
  schröder_cols = c("german", "translation", "semantic category", "generation nb total", "generation % total", "typicality mean", "typicality SD", "S: AoALikert mean", "S: AoALikert SD", "familiarity mean", "familiarity SD", "DLEXDB normalized lemma freq per million", "DLEXDB normalized log10 lemma freq", "nb phonemes", "nb syllables")
  schröder_norms <- read_excel('../../external_resources/norms/Schröder_2012.xls', col_names = schröder_cols, skip=2)
  # filter for useful columns
  schröder_norms <- schröder_norms %>% 
    mutate(NAME1 = remove_umlauts(german)) %>% # lowercase words + remove umlauts
    mutate(NAME1 = ifelse(NAME1 == "chamaeleon", "chameleon", NAME1)) # account for "Chamäleon" being written incorrectly in MultiPic
  if(select_cols==TRUE){
    schröder_norms <- schröder_norms %>% 
      select(NAME1, "S: AoALikert mean", "S: AoALikert SD")
  }
  return(schröder_norms)
}

kuperman_correlations <- function(df){
  # get list of unique (remaining) subject IDs
  ids <- unique(df$ID)
  # create overview table for participants' correlations with Birchenough et al. norms
  shared_correlations <- tibble("ID" = ids, corr = 0)
  # select only shared items that have a rating in Birchenough et al.
  kuperman_procedure_shared <- df[df$item_number %in% shared_items & !is.na(df$`B: AoA mean`),]
  # calculate participants' correlations of shared items with Birchenough norms and add to correlation overview
  for (id in ids) {
    sub_shared <- kuperman_procedure_shared[kuperman_procedure_shared$ID == id,]
    correlation <- with(sub_shared, cor(estimate, `B: AoA mean`))
    shared_correlations[shared_correlations$ID == id,]$corr <- correlation
  }
  # weak correlations: correlations < 0.4
  weak_corr_ids <- shared_correlations[shared_correlations$corr < 0.4,]$ID
  return(list("corr_df" = shared_correlations, "weak_corr_ids" = weak_corr_ids))
}

birchenough_correlations <- function(df){
  # get list of unique (remaining) subject IDs
  ids <- unique(df$ID)
  # create overview table for participants' mean control word ratings (CWMs) correlations with mean group CWMs
  cwm_correlations <- tibble("ID" = ids, corr = 0)
  # select only shared items (with participants still present and estimates != NA)
  birchenough_procedure_shared <- df[df$item_number %in% shared_items & !is.na(df$estimate),]
  # get group estimates
  prelim_aoa_estimates <- get_estimates_overview(birchenough_procedure_shared, all=FALSE)
  for (id in ids) {
    sub_shared <- birchenough_procedure_shared[birchenough_procedure_shared$ID == id,]
    # limit group estimates to entries present in sub_shared
    relevant_group_estimates <- prelim_aoa_estimates[prelim_aoa_estimates$item_number %in% sub_shared$item_number,]
    # correlate
    correlation <- cor(sub_shared$estimate, relevant_group_estimates$estimate_mean)
    cwm_correlations[cwm_correlations$ID == id,]$corr <- correlation
  }
  weak_cwm_corr_ids <- cwm_correlations[cwm_correlations$corr < 0.4,]$ID
  return(list("corr_df" = cwm_correlations, "weak_corr_ids" = weak_cwm_corr_ids))
}

within_participant_corelations <- function(df){
  # Calculate participants' reliability by means of repeated item ratings.
  # get list of unique (remaining) subject IDs
  ids <- unique(df$ID)
  # create overview table for participants' correlations within repeated items
  repeated_correlation <- tibble("ID" = ids, corr = 0)
  # select only repeated items (with estimates != NA)
  full_repeated <- df[df$item_number %in% repeated_items & !is.na(df$estimate),]
  for (id in ids) {
    sub_repeated_0 <- full_repeated[full_repeated$ID == id & full_repeated$repetition == 0,]
    sub_repeated_1 <- full_repeated[full_repeated$ID == id & full_repeated$repetition == 1,]
    # check if there are estimates for both repeated items
    missing_item <- symdiff(c(sub_repeated_0$item_number), c(sub_repeated_1$item_number))
    if (length(missing_item) == 0) {
      # if yes: correlate
      correlation <- cor(sub_repeated_0$estimate, sub_repeated_1$estimate)
    } else {
      # if not: exclude incomplete item set from correlation calculation
      sub_repeated_0 <- sub_repeated_0 %>% filter(!item_number %in% missing_item)
      sub_repeated_1 <- sub_repeated_1 %>% filter(!item_number %in% missing_item)
      correlation <- cor(sub_repeated_0$estimate, sub_repeated_1$estimate)
    }
    repeated_correlation[repeated_correlation$ID == id,]$corr <- correlation
  }
  weak_participant_ids <- repeated_correlation[repeated_correlation$corr < 0.4,]$ID
  return(list("corr_df" = repeated_correlation, "weak_corr_ids" = weak_participant_ids))
}

plot_corr_distribution <- function(corr_df, id_list, title, caption){
  # Helper function to plot the distribution of correlations
  corr_df %>% 
    mutate(exclusion = if_else(ID %in% id_list, "yes", "no")) %>% 
    ggplot(aes(x=corr, fill = exclusion)) + 
    geom_histogram(bins = 30) +
    geom_vline(aes(xintercept = mean(corr_df$corr)), linetype=2) +
    #geom_text(data = annotations, aes(x = corr, y = y, label = label))
    scale_fill_brewer(palette= "Paired") +
    labs(title = title, caption = caption) +
    theme_minimal()
}

save_corr_distribution_plot <- function(corr_df, id_list, save_path){
  corr_plot <- corr_df %>% 
    mutate(exclusion = if_else(ID %in% id_list, "yes", "no")) %>% 
    ggplot(aes(x=corr, fill = exclusion)) + 
    geom_histogram(bins = 30) +
    geom_vline(aes(xintercept = mean(corr_df$corr)), linetype=2) +
    scale_fill_brewer(palette= "Paired") +
    theme_minimal()
  save_plot(save_path, corr_plot)
}