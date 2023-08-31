# Materials Optimization Algorithm and Connecting to FactSage Software MACRO

This repository contains code for an optimization algorithm that utilizes a genetic algorithm approach to optimize a material composition for desired mechanical properties. The algorithm evolves a population of material compositions over multiple generations, using a random forest regression model to evaluate their fitness.

## Overview

The optimization algorithm consists of the following main components:

1. Data Preparation:
   - Definitions of different second phases main element in perioodic table (Alkline, Transition, Lanthanides, Post_transition, Metalloids).
   - Data processing steps to aggregate composition data and calculate the sum of different second phases and adding to the dataset.
   - Splitting the data into input features and target variables.

2. Model Training and Evaluation:
   - Training a random forest regression model to predict mechanical properties based on material compositions.
   - Evaluating the model performance using metrics such as mean absolute error (MAE), mean squared error (MSE), and R2 score.

3. Genetic Algorithm Optimization:
   - Creating an initial population of material compositions.
   - Evolving the population through selection, crossover, and mutation operations.
   - Evaluating the fitness of each composition using the random forest regression model.
   - Generating new compositions by running an external program and extracting composition data.
   - Storing the best compositions and their fitness values for analysis.

4. Analysis and Visualization:
   - Plotting the fitness values of the best compositions over the generations.
   - Visualizing the fitness values of specific material categories.
   - Identifying the best solution and its fitness value.
   - Saving the best solution as a DataFrame.


## Contributing

We welcome contributions to enhance the optimization algorithm. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request describing your changes.

## Contact

For questions, issues, or collaboration opportunities, please contact [Parham Valipoorsalimi] at [Parham.valipoorsalimi@mail.mcgill.ca].
