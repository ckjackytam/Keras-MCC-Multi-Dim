def matthews_correlation(y_true, y_pred):
    # Calculate the MCC for multi-class classification
    #Calculate the total number of samples
    s = K.sum(y_true)
    #Calculate the number of times each class occurred
    t = K.sum(y_true, axis=0)
    #Allocate the predicted class based on the max probability
    predict_argmax = K.argmax(y_pred, axis=1)
    #Create a one hot matrix based on the predicted class
    encoded = K.one_hot(predict_argmax, 3)
    #Calculate the number of times each class was predicted
    p = K.sum(encoded, axis=0)
    #Calculate the product of the actual class vector and predicted class vector
    actual_predict = t * p
    #Square the actual class vector and the predicted class vector
    t_sq = K.square(t)
    p_sq = K.square(p)
    #Calculate the total no. of samples correctly predicted
    c = K.sum(y_true * encoded)
    #Calculate the numerator and the denominator
    numerator = c * s - K.sum(actual_predict)
    denominator = K.sqrt((K.square(s) - K.sum(p_sq)) * (K.square(s) - K.sum(t_sq)))
    # Calculate the multi-class MCC
    MCC = numerator / (denominator + K.epsilon())
    return MCC