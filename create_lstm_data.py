import numpy as np
 
def create_lstm_data(data, lag = 1):
    """
    Creates features and targets suitable for LSTM modelling
    See https://keras.io/api/layers/recurrent_layers/lstm/ for more information
    
    Inputs
    ------
    data: np.array
        One dimensional time series np.array to be processed
        This array should not have undergone the process: .reshape(-1, 1)
    lag: int
        Amount of historical lag used to create the feature set
    
    Outputs
    -------
    X, y: np.array, np.array
        Feature and target arrays
    """
    if lag > len(data):
        # lag cannot be more than data length
        return None
    
    if len(np.shape(data)) > 1 and np.shape(data)[1] == 1:
        # if for some reason data was detected to have undergone reshape(-1, 1)
        # force it back to a standard 1 dimensional np.array
        data = data.reshape(-1)
    elif len(np.shape(data)) > 1 and np.shape(data)[1] > 1:
        # this function works only for 1D arrays. Anything more than that will not work
        return None
        
    # if there are no pressing issues, create the LSTM compatible data 
    # with the specified historical lag
    X = np.zeros([len(data) - lag, lag])
    y = np.zeros(len(data) - lag)
    for n in range(len(data) - lag):
        X[n] = data[n:n + lag]
        y[n] = data[n + lag]
        
    # X needs to be a 3D tensor with shape: [batch, timesteps, feature]
    X = X.reshape([X.shape[0], X.shape[1], 1])
    
    return X, y

def demo(lag = 3):
    # demonstrates how to use create_lstm_data()
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 ,18 ,19, 20])
    X, y = create_lstm_data(X, lag)

    for i in range(len(X)):
        print(X[i], y[i])
