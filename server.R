# Instantiate our N-Gram Prediction Engine:
source("./predictionEngine.R")

########################################################################################################################
#       
#       This function handles text input from the web client
#       It takes the desired text and runs it through the predition engine and renders text in the web client
#       @param input    String
#       @param output   String
#       @return Null
#
########################################################################################################################
shinyServer(function(input, output) 
{
        
        # Reactive listens to the web client for input change and 'reacts':
        wordPrediction <- reactive({
                # Get the new text input:
                text <- input$text
                # Clean it for prediction purposes:
                textInput <- cleanInput(text)
                # Get the length of the string:
                wordCount <- length(textInput)
                # Run the text input through the N-Gram Prediction engine:
                wordPrediction <- nextWordPrediction(wordCount,textInput)})
        
        # Tell the web client to output the predicted word:
        output$predictedWord <- renderPrint(wordPrediction())
        # Also tell the web client to print what was entered as a sanity check:
        output$enteredWords <- renderText({ input$text }, quoted = FALSE)
})