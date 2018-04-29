# psoAutoML
To run the program in serial, run:

> python pso_parallel.py

To run the program in parallel:

> python -m scoop pso_parallel.py 

To run the program in parallel on specified hosts:

> python -m scoop --hostfile \<hostfilepath\> pso_parallel.py 

# Current hardcoded components 
In pso_parallel.py, the hyperparameters are assumed to be in the order of 
1. number of convolution filters 
2. number of dense outputs 
3. kernel size
4. pooling size
5. drop out rate

The bd_min (minimum boundary condition) and the bd_max (maximum boundary condition) needs to be initialised in the same order.

If any of these parameters are to be removed, the global constance DIM needs to be changed to match the number of hyperparameters. 
The boundary conditions and the neuralnet evaluation function in the file 'evaluation_functions.py' will also need to be modified to match the new hyperparamter setting/selection.

# Neural network file
These parameters need to be modified with new datasets
1. num_channels - number of colour channels
2. num_rows - expected height of the images
3. num_cols - expected width of the images 
4. num_classes - expected number of classifications 

-- Need to include stuff about the importing of datasets (np array shaping and such) -- 
