import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import math

data = pd.read_csv('amazon_stock_data_max.csv')   
data['Open'] = data['Open'].str.replace('$', '').astype(float)
open_data = data['Open'].to_list()
open_data_stream = data['Open']

# Method to initialise all weight matrices to zero
def initialize_weights(input_size, hidden_size, output_size, num_students):
    """
    Initialize the weights for the neural network.

    Parameters:
        input_size (int): Number of input neurons.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons.
        num_students (int): Number of student models.

    Returns:
        dict: Dictionary containing weight matrices for the neural network.
    """
    weights = {
        'V': np.zeros((num_students, hidden_size, input_size)),
        'W': np.zeros((num_students, output_size, hidden_size))
    }
    return weights

# Method to standardize data
def standardize_data(data):
    """
    Standardize the input data using StandardScaler.

    Parameters:
        data (list or array): Input data to be standardized.

    Returns:
        array: Standardized input data.
    """
    # Reshape data to 2D array with a single column
    data_2d = np.array(data).reshape(-1, 1)
    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data_2d).flatten()
    return standardized_data

# Method to multiply the weights with the input layer in order to compute the hidden layer 
def compute_hidden_layer(weights, input_layer):
    """
    Compute the hidden layer by multiplying the weights with the input layer.

    Parameters:
        weights (dict): Neural network weights.
        input_layer (array): Input layer values.

    Returns:
        array: Hidden layer values.
    """
    return np.dot(weights['V'], input_layer)

# Method to compute the output of a neural network
def compute_output(weights, input_layer):
    """
    Compute the neural network output.

    Parameters:
        weights (dict): Neural network weights.
        input_layer (array): Input layer values.

    Returns:
        float: Neural network output.
    """
    # Compute the hidden layer
    hidden_layer = compute_hidden_layer(weights, input_layer)
    # Multiply the hidden layer with its weights leading to the final output
    output = np.dot(weights['W'], hidden_layer)
    return output.item()

# Method to compute the difference between the teacher's output and the student's output
def compute_residual(teacher_output, student_output):
    """
    Compute the difference between the teacher's output and the student's output.

    Parameters:
        teacher_output (float): True output from the teacher model.
        student_output (float): Predicted output from the student model.

    Returns:
        float: Residual between teacher and student outputs.
    """
    
    return math.dist([teacher_output], [student_output])

# Method to compute the difference between the true output and the teacher's output
def compute_shift(teacher_output, true_output):
    """
    Compute the difference between the true output and the teacher's output.

    Parameters:
        teacher_output (float): True output from the teacher model.
        true_output (float): True output from the dataset.

    Returns:
        float: Shift between teacher and true outputs.
    """

    return math.dist([true_output], [teacher_output]) 

# Method to compute the empirical generalisation error as the Mean Squared Error of the residuals
def estimate_mse(residual):
    """
    Estimate Mean Squared Error (MSE) from residuals.

    Parameters:
        residual (float): Residual between true and predicted outputs.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean(np.square(residual))

# Method to compute the teacher's weights
def estimate_teacher_weight_vector(weights, num_students, rotation_factor = 0.75):
    """
    Estimate the teacher's weight vector.

    Parameters:
        weights (dict): Neural network weights of student models.
        num_students (int): Number of student models.
        rotation_factor (float): Factor to introduce non-linearity.

    Returns:
        dict: Estimated teacher's weight vector.
    """
    
    # Taking the sum of weights (respectively) of all students
    summed_weights = {key: np.sum(value, axis=0) for key, value in weights.items()}
    # Normalizing the summed weights by dividing them by thier norm
    normalized_weights = {key: value / np.linalg.norm(value) for key, value in summed_weights.items()}
    '''
    Rotating the weights by a factor of 0.75 to better match the true output
    Since our model is entirely linear, it does not allow non-linear transformations which are necessary for accurate regression
    Hence, we introduce non-linearity by manually multiplying the teacher's weights by a rotation_factor
    in order to make the output better fit the true output.
    '''
    rotated_weights = {key: rotation_factor * value for key, value in normalized_weights.items()}
    return rotated_weights

def estimate_teacher_pdf(teacher_weights, grid_size=16):
    """
    Estimate the probability density function (PDF) of the teacher.

    Parameters:
        teacher_weights (dict): Teacher's weight vector.
        grid_size (int): Size of the grid for PDF estimation.

    Returns:
        tuple: Tuple containing the estimated PDF and corresponding weight grid.
    """
    # Computing the grid for PDF estimation
    bk = teacher_weights['W'].flatten()
    b_max = np.square(max(bk))
    b_grid = b_max * np.linspace(-1, 1, num=2**(grid_size))
    
    # Computing the PDF by taking the Fast Fourier Transform (FFT) of the characteristic function
    PSI = []
    for bi in b_grid:
        # Evaluating the characteristic function assuming a normal distribution
        psi = np.exp((bi**2) / 2)
        PSI.append(psi)
    # Computing the FFT of the characteristic function
    P = fft(PSI)
    return np.real(P), b_grid

def loss_function(b,R):
    """
    Compute the loss function value.

    Parameters:
        b (float): Weight value.
        R (float): Overlap parameter.

    Returns:
        float: Loss function value.
    """
    return (b - b*R)**2

def gamma(b, R):
    """
    Compute the gamma function.

    Parameters:
        b (float): Weight value.
        R (float): Overlap parameter.

    Returns:
        float: Gamma function value.
    """
    mean = 0
    variance = 1 - R**2
    pdf_value = norm.pdf(b * R, loc=mean, scale=variance)
    result = b * pdf_value
    return result
                    
def update_overlap_parameter(R, P, e_m, b):
    """
    Update the overlap parameter using the formula in the OPE algorithm.

    Parameters:
        R (float): Current overlap parameter.
        P (array): Probability density function.
        e_m (float): Empirical generalization error.
        b (array): Weight grid.

    Returns:
        float: Updated overlap parameter.
    """
    delta = 0.1
    diff = 1
    while diff > delta:
        R_new = R + (1 - R**2) * ((np.dot(P, loss_function(b,R))) - e_m) / (np.dot(P, gamma(b,R)))
        diff = R_new - R
    return R_new
                         
def nu(b, R, residual):
    """
    Compute the nu function.

    Parameters:
        b (float): Weight value.
        R (float): Overlap parameter.
        residual (float): Residual value.

    Returns:
        float: Nu function value.
    """
    mean = b*R
    variance = abs(1 - R**2)
    pdf_value = norm.pdf(residual, loc=mean, scale=variance)
    return pdf_value

def estimate_teacher_post_synaptic_field(P, residual, b, R):
    """
    Estimate the teacher's post-synaptic field.

    Parameters:
        P (array): Probability density function.
        residual (float): Residual value.
        b (array): Weight grid.
        R (float): Overlap parameter.

    Returns:
        float: Estimated post-synaptic field value.
    """
    Nu = nu(b, R, residual)
    Up = b * Nu
    bi = np.real(P.T @ Up) / np.real(P.T @ Nu)
    return bi

def update_learning_rate(weights, R, bi, phi):
    """
    Update the learning rate for weight updates.

    Parameters:
        weights (dict): Neural network weights.
        R (float): Overlap parameter.
        bi (float): Estimated post-synaptic field.
        phi (float): Residual value.

    Returns:
        tuple: Tuple containing updated learning rates for V and W layers.
    """
    V_size = np.prod(weights['V'].shape)
    W_size = np.prod(weights['W'].shape)
    Qv =  np.sum(np.square(weights['V'])) / V_size 
    Qw =  np.sum(np.square(weights['W'])) / W_size
    Fv = (np.sqrt(Qv) / np.abs(R)) * (bi - (R * phi))
    Fw = (np.sqrt(Qw) / np.abs(R)) * (bi - (R * phi))
    return Fv, Fw

def update_weights(weights, norm_residual, input_layer, Fv, Fw):
    """
    Update the neural network weights.

    Parameters:
        weights (dict): Neural network weights.
        norm_residual (float): Normalized residual value.
        input_layer (array): Input layer values.
        Fv (float): Learning rate for V layer.
        Fw (float): Learning rate for W layer.

    Returns:
        dict: Updated neural network weights.
    """
    V_size = np.prod(weights['V'].shape)
    W_size = np.prod(weights['W'].shape)
    input_layer = np.array(input_layer)
    weights['V'] += Fv * (norm_residual * input_layer) / np.sqrt(V_size)
    hidden_layer = compute_hidden_layer(weights, input_layer)
    weights['W'] += Fw * (norm_residual * hidden_layer) / np.sqrt(W_size)
    return weights

def ope_algorithm(num_students, input_size, hidden_size, output_size, data):
    """
    Implement the Online Prediction Error (OPE) algorithm.

    Parameters:
        num_students (int): Number of student models.
        input_size (int): Number of input neurons.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output neurons.
        data (list or array): Input data for training.

    Returns:
        tuple: Tuple containing teacher weights, MSE, shifts, mean_shifts, teacher_outputs, and true_outputs.
    """
    
    # Initialisation
    input_data = np.log(data / data.shift(1)).dropna()
    seed_layer = [1,1,1]
    teacher_output = 1.0
    mse_error = 0.5
    R = 0
    
    MSE = []
    shifts = []
    mean_shifts = []
    teacher_outputs = []
    true_outputs = []
    
    # Initialise weights
    student_weights = initialize_weights(input_size, hidden_size, output_size, num_students)
    teacher_weights = initialize_weights(input_size, hidden_size, output_size, 1)
    
    # Run the zeroeth iteration of the algorithm in order to set it up.
    first_residuals = []
    
    # Feedforward
    for i in range(num_students):
        # get weights of student i
        weights_student_i = {key: value[i] for key, value in student_weights.items()}
        # compute output of student i
        student_output = compute_output(weights_student_i, seed_layer)
        # compute the residual of student i
        residual = compute_residual(teacher_output, student_output)
        first_residuals.append(residual)
    
    # Backpropagation
    for i in range(num_students):
        # get weights of student i
        weights_student_i = {key: value[i] for key, value in student_weights.items()}
        # compute the norm of the student's residual
        norm_residual = first_residuals[i] / np.linalg.norm(first_residuals)
        # update the weights of student i
        new_weights = update_weights(weights_student_i, norm_residual, seed_layer, Fv=1, Fw=1)
        for key, value in student_weights.items():
            value[i] = new_weights[key]
    
    # Estimate the teacher's weights for the first iteration
    teacher_weights = estimate_teacher_weight_vector(student_weights, num_students)
    
    # Iterative Algorithm
    for p in range(1, np.size(input_data)-2):
        
        # Initialise variables
        input_layer = [input_data[p], input_data[p+1], input_data[p+2]]
        student_outputs = []
        residuals = []
        true_output = input_data[p+2] # The true output of the previous iteration is the last input value of the next iteration
        true_outputs.append(true_output)
        shift = compute_shift(teacher_output, true_output) # Compute the shift between the last output of our model and it's true output
        shifts.append(shift)
        mean_shift = np.mean(shifts)
        mean_shifts.append(mean_shift)
        
        print('p: ', p)
        print('input: ', input_layer)
        print('mean_shift: ', mean_shift)
        
        # Compute the teacher's output using old weights from the previous iteration
        teacher_output = compute_output(teacher_weights, input_layer)
        teacher_outputs.append(teacher_output)
        
        print('output: ', teacher_output)
        
        # Feedforward
        for i in range(num_students):
            # get weights of student i
            weights_student_i = {key: value[i] for key, value in student_weights.items()}
            # compute output of student i
            student_output = compute_output(weights_student_i, input_layer)
            student_outputs.append(student_output)
            # compute the residual of student i
            residual = compute_residual(teacher_output, student_output)
            residuals.append(residual)
        
        print('so: ', np.mean(student_outputs))
        print('sw: ', student_weights)
            
        # Backpropagation
        # compute the empirical error between the student's output and the teacher's.
        mse_error = estimate_mse(residuals)
        MSE.append(mse_error)
        # estimate the teacher's new weights by averaging over the student's weights
        teacher_weights = estimate_teacher_weight_vector(student_weights, num_students)
        # compute the pdf and grid_vector of the teacher
        pdf, b = estimate_teacher_pdf(teacher_weights)
        # compute the updated value for R
        R = 0
        R = update_overlap_parameter(R, pdf, mse_error, b)
        
        print('MSE: ', mse_error) 
        
        # Update
        for i in range(num_students):
            # compute the estimate for the teacher's post-synaptic field according to student i
            bi = estimate_teacher_post_synaptic_field(pdf, residuals[i], b, R)
            # get weights of student i
            weights_student_i = {key: value[i] for key, value in student_weights.items()}
            # compute the adapted learning rates Fv and Fw for student i
            Fv, Fw = update_learning_rate(weights_student_i, R, bi, residuals[i])
            # compute the norm of the student's residual
            norm_residual = residuals[i] / np.linalg.norm(residuals)
            # update the weights of student i
            new_weights = update_weights(weights_student_i, norm_residual, input_layer, Fv, Fw)
            for key, value in student_weights.items():
                value[i] = new_weights[key]
            
    return teacher_weights, MSE, shifts, mean_shifts, teacher_outputs, true_outputs

# Example usage
num_students = 10
input_size = 3
hidden_size = 6
output_size = 1

# Assuming you have input_data and teacher_output
weights_1, MSE_1, shifts_1, mean_shifts_1, teacher_outputs_1, true_outputs_1 = ope_algorithm(num_students, input_size, hidden_size, output_size, open_data_stream)
