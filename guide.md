# These are the steps to follow to create working pipeline for the project


## 1. *Data preparation*
    1. Load the raw data and the original PSS/SSS sequence
        - The PSS/SSS sequence is as follow: SSS1, PSS1, SSS2, PSS1 and then repeats 4 times (thus length of 16)
    2. Average the PSS/SSS sequence by "folding" it 
    3. Use the averaged PSS/SSS sequence (now length 4) to estimate the channel response. 
        -   This is done by tiling the PSS/SSS sequence in the first axis to match the length of the raw signal and 
            then diving the raw signal by the tiled PSS/SSS sequence
    4. Calculate the magniture of the channel reposne in dB
    5. Stack all channel responses to form a matrix of size (72, N)
    6. Save these matrices in a form of torch tensor (.pt file)

## 2. *Training*
    1. Load the channel response matrices
    2. Split the data into training and validation sets
    3. Create a model
    4. Train the model
        - The truth values is the input value of the channel response
        - The reconstruction error (either RMSE or MSE) is the different between the output of the model and the truth value
    5. Save the model

## 3. *Testing*
    1. Load the model
    2. Load the channel response matrices of different dataset (with and without fake BTS but not the training data)
    3. Validate the models performance on fake and true BTS and record mean error
    4. Use the average the mean error of true and fake BTS as the threshold for the model (or just eye ball it)
    5. Use the threshold to classify the BTS as true or fake
    6. Review the results and adjust the threshold if necessary
    5. Save the model and record its architecture into a shared document 
        - All training parameters, model type, true and fake BTS error, threshold (final), classification accuracy, etc.
        - Save the model using torh.save(model) (save the whole model, not just the dict) 
        - In case of custom layers and building block record the models architecture in a separate file


