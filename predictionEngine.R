# Install the necessary packages;
suppressPackageStartupMessages(c(
        library(shinythemes),
        library(shiny),
        library(tm),
        library(stringr),
        library(markdown),
        library(stylo)))

# Load our data corpus as RDS files:
trigramData <- readRDS(file="./data/trigramData.RData")
bigramData <- readRDS(file="./data/bigramData.RData")
unigramData <- readRDS(file="./data/unigramData.RData")

################################################################################
#       
#       This function takes text data and cleans it for N-Gram Word Prediction
#       @param text String
#       @return String
#
################################################################################
cleanTextStrInput <- function(textStr)
{
        # Run textStr input through a gauntlet of cleaning operations:
        cleanedTextStr = tolower(textStr)
        cleanedTextStr = removePunctuation(cleanedTextStr)
        cleanedTextStr = removeNumbers(cleanedTextStr)
        cleanedTextStr = str_replace_all(cleanedTextStr, "[^[:alnum:]]", " ")
        cleanedTextStr = stripWhitespace(cleanedTextStr)
        
        # Now return the cleaned string:
        return(cleanedTextStr)
}

##################################################################################################
#       
#       This function takes cleaned text input and converts everything to the English language
#       @param text String
#       @return String
#
##################################################################################################
cleanInput <- function(textStr)
{
        #  Clean Text:
        textInput = cleanTextStrInput(textStr)
        textInput = txt.to.words.ext(textInput, language="English.all", preserve.case = TRUE)
        
        # Return cleaned text:
        return(textInput)
}

##################################################################################################
#       
#       This function uses the N-Gram algorithm to predict the next word in a given sequence of words
#       @param  wordCount Integer
#       @param  textInput String      
#       @return String
#
##################################################################################################
nextWordPrediction <- function(wordCount,textInput)
{
        # First step is to figure out how many words we have 
        if (wordCount>=3) 
        {
                # If we have more than three words, cull the first two
                textInput <- textInput[(wordCount-2):wordCount] 
        }
        else if(wordCount==2) 
        {
                # If we have exactly two words create a vector with the first value being NULL
                textInput <- c(NA,textInput)   
        }
        else 
        {
                # Otherwise we have one word:
                textInput <- c(NA,NA,textInput)
        }
        
        # First crack at getting the word:
        unigramInput = textInput[1]
        bigramInput = textInput[2]
        trigramInput = textInput[3]
        
        # First try to return a quadram:
        predictedWord <- as.character(trigramData[trigramData$unigram==unigramInput & trigramData$bigram==bigramInput & trigramData$trigram==trigramInput,][1,]$quadgram)
        
        # If our prediction function failed to return anything:
        if(is.na(predictedWord)) 
        {
                # Try to return a trigram:
                predictedWord <- as.character(bigramData[bigramData$unigram==bigramInput & bigramData$bigram==trigramInput,][1,]$trigram)

                # And ... if that doesn't work:
                if(is.na(predictedWord)) 
                {
                        # Try to return a bigram, otherwise we return NA:
                        predictedWord <- as.character(unigramData[unigramData$unigram==textInput[3],][1,]$bigram)
                }
        }
        
        # Return the predicted word to the web client:
        print(predictedWord)
}