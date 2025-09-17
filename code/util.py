import numpy as np
from ollama import embed
from scipy import stats
import pandas as pd

def get_embedding(input: str, model: str):
    '''
    return embeding of a word or a sentence
    
    Args:
        input(str): a word or a sentence
        model(str): name of the model
    Return:
        normalized ndarray of shape (1,size_of_embedding) and dtype float 64 of shape (1,1) 
    '''
    return np.array(embed(model=model, input=input)['embeddings']) 


def cosine_similarity(u, v):
    '''
    calculate the cosine similariy between two vector

    Args:
        u(ndarray):
        v(ndarray):

    Return:
        floating number for their cosine similarity

    '''
    return np.dot(u, v.T)/(np.linalg.norm(u)*np.linalg.norm(v))


# analyze results
def normalize(u):
    '''
    Args: 
        u (ndarray): intended for column or row vector

    Returns:
        float: normalized array
    '''
    mean = np.mean(u)
    std = np.std(u)
    return (u -mean) / std

def standardize(u):
    '''
    Args:
        u (ndarray): intended fr column or row vector
    Return:
        standardize ndarray  

    '''
    MIN, MAX= u.min() , u.max()
    standard = (u - MIN)/(MAX - MIN)
    return standard

def rmse(actual_values, predicted_values):
    """
    Calculates the Root Mean Squared Error (RMSE) between actual and predicted values.

    Args:
        actual_values (np.array): A NumPy array of actual values.
        predicted_values (np.array): A NumPy array of predicted values.

    Returns:
        float: The calculated RMSE value.
    """
    # Calculate the differences
    differences = actual_values - predicted_values
    
    # Square the differences
    squared_differences = np.square(differences)
    
    # Calculate the mean of the squared differences (MSE)
    mse = np.mean(squared_differences)
    
    # Take the square root to get RMSE
    rmse = np.sqrt(mse)

    return rmse

def round_to(input,round_percision):
    '''
        round to values like 0.25 or 0.5

    Args:
        input (float): 
        round_percision (round_percision): something like 0.25 or 0.5

    Returns:
        result(float): 
    '''

    inverse = 1/round_percision
    
    return round(input*inverse) / inverse



def print_scores(ground_truth, predictions):
    '''
    given the actual resault and prediction calculate the rmse and pearson r

    Args:
        ground_truth (ndarray or pandas Series):  
        predictions (ndarray or pandas Series): should be between 1 and 5 in most cases

    '''
    #if ( type(ground_truth) == type(predictions) == type(pd.Series())):
    if type(ground_truth) == type(pd.Series()):
        ground_truth = ground_truth.to_numpy().reshape(-1,1)

    if type(predictions) == type(pd.Series()):   
        predictions = predictions.to_numpy().reshape(-1,1)

    assert predictions.shape == ground_truth.shape

   
    print('rmse: ', rmse(ground_truth,predictions) , end=' ')
    print('r: ', stats.pearsonr(ground_truth.reshape(-1), predictions.reshape(-1)).statistic)


from tqdm import tqdm

def answers_embedded_y(data, model = 'mxbai-embed-large', size_of_embedding=1024):
    '''
    given a subset(of rows) of the answers from the main dataframe and the specified embedding 
    retrun X: embedding of student answer and y and the score_avg

    
    note tom my self: X is the same as student answers embedded correct answers are commented intentionaly
    this will be used when trying to fit onky only on student answers embeddings


    Args:
    data: pandas datafram
    model: 


    return:
    X = ndarray of size (number of sample,size_of_embedding)
    y = ndarray of size (number of sample,1)
    '''

    # correct_answers = data.iloc[:,1].to_numpy().reshape(-1,1)
    student_answers = data.iloc[:,2].to_numpy().reshape(-1,1)


    size_of_embedding = 1024 #768
    # correct_answers_embedded = np.zeros(shape=(correct_answers.shape[0],size_of_embedding))
    X = np.zeros(shape=(student_answers.shape[0],size_of_embedding))
    for i in tqdm(range(X.shape[0])):
        
        # correct_answers_embedded[i] = util.get_embedding(input=correct_answers[i,0], model=model).reshape(-1)
        X[i] = get_embedding(input=student_answers[i,0], model=model).reshape(-1)
        # dot_score_matrice = np.dot(correct_answers_embedded , student_answers_embedded.T)

    y = data['score_avg'].to_numpy().reshape(-1,1)

    return X, y     

def get_E(series, model, size_of_embedding):
    '''
    return embeddings for a series of sentences
    Args:
        series (pandas series): containint senetences

        
    Returns:
        np.array: shape is (N_rows,embedding_size)

    '''
    series = series.to_numpy().reshape(-1,1)
    E = np.zeros(shape=(series.shape[0],size_of_embedding))
    for i in tqdm(range(series.shape[0])):

        E[i] = get_embedding(input=series[i,0], model=model).reshape(-1)
    return E

def get_Y(df, y_col='score_avg'):
    '''
    return the Y given the entire dataframe as a numpy array

    Args:
        series (pandas series): containint senetences
        y_col: default name for the column 
        
    Returns:
        np.array: shape is (N_rows,1)

    '''
    return df[y_col].to_numpy().reshape(-1,1)


def get_Y_scaled(df, y_col='score_avg'):
    '''
    return the Y (scaled from 0.0 to 1.0) given the entire dataframe as a numpy array

    Args:
        series (pandas series): containint senetences
        y_col: default name for the column 
        
    Returns:
        np.array: shape is (N_rows,1)

    '''
    return (df[y_col].to_numpy().reshape(-1,1) / 4) - 0.25

def calculate_dot_score(desired_answers_E, student_answers_E, scale_to_5 = False):
    '''
    give the dot score similarty.

    Args:
        student_answers_E(np.array): shape = (N_rows,embedding_size)
        desired_answers_E(np.array): shape = (N_rows,embedding_size)
        scale_to_5(bool): decide if the range is [0,1] or [1,5]
    Returns:
        np.array: similarities of shape = (N_rows,1). if scale_to_5 ==  True then give final score

    '''
    dot_scores  = np.diag(np.dot(desired_answers_E , student_answers_E.T)).reshape(-1,1)
    if scale_to_5:  
        return (dot_scores*4 ) +1
    return dot_scores
    
