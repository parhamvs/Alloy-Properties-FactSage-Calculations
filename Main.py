import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import plot_partial_dependence
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import math
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
import random
import time
import xlsxwriter

def select_mating_pool(pop, fitness, num_parents):
    """
    Selects the best individuals in the current generation as parents for producing the offspring of the next generation.

    Args:
        pop (numpy.ndarray): Current population.
        fitness (numpy.ndarray): Fitness values for each individual in the population.
        num_parents (int): Number of parents to select.

    Returns:
        numpy.ndarray: Selected parents.
    """
    parents = np.empty((num_parents, pop.shape[1]))
    
    for parent_num in range(num_parents):
        max_fitness_idx = np.argmax(fitness)  # Find the index of the individual with the maximum fitness
        parents[parent_num, :] = pop[max_fitness_idx, :]  # Select the individual as a parent
        fitness[max_fitness_idx] = -np.inf  # Mark the selected individual with a low fitness value
    
    return parents


def crossover(parents, offspring_size):
    """
    Performs crossover operation between parents to produce offspring.

    Args:
        parents (numpy.ndarray): Selected parents for crossover.
        offspring_size (tuple): Size of the offspring array.

    Returns:
        numpy.ndarray: Offspring generated through crossover.
    """
    offspring = np.empty(offspring_size)
    crossover_point = offspring_size[1] // 2  # Crossover point at the center
    
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]  # Index of the first parent to mate
        parent2_idx = (k + 1) % parents.shape[0]  # Index of the second parent to mate
        
        # Inherit genes from parents to create offspring
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    
    return offspring
 

def mutation(offspring_crossover):
    """
    Applies mutation by changing a single gene in each offspring randomly.

    Args:
        offspring_crossover (numpy.ndarray): Offspring after crossover.

    Returns:
        None.
    """
    limits = [3, 3, 8, 0.75, 0, 0, 3, 0, 0, 0, 0]
    random_value1 = np.random.normal(-0.1, 0.1, 7)
    random_value2 = np.random.normal(-0.5, 0.5, 5)
    
    for idx in range(offspring_crossover.shape[0]):
        if 0 < offspring_crossover[idx, 0] + random_value1[0] < limits[0]:
            offspring_crossover[idx, 0] += random_value1[0]
        if 0 < offspring_crossover[idx, 1] + random_value1[1] < limits[1]:
            offspring_crossover[idx, 1] += random_value1[1]
        if 0 < offspring_crossover[idx, 2] + random_value2[0] < limits[2]:
            offspring_crossover[idx, 2] += random_value2[0]
        if 0 < offspring_crossover[idx, 3] + random_value1[2] < limits[3]:
            offspring_crossover[idx, 3] += random_value1[2]
        if 0 < offspring_crossover[idx, 4] + random_value1[3] < limits[4]:
            offspring_crossover[idx, 4] += random_value1[3]
        if 0 < offspring_crossover[idx, 5] + random_value2[1] < limits[5]:
            offspring_crossover[idx, 5] += random_value2[1]
        if 0 < offspring_crossover[idx, 6] + random_value1[4] < limits[6]:
            offspring_crossover[idx, 6] += random_value1[4]
        if 0 < offspring_crossover[idx, 7] + random_value2[2] < limits[7]:
            offspring_crossover[idx, 7] += random_value2[2]
        if 0 < offspring_crossover[idx, 8] + random_value1[5] < limits[8]:
            offspring_crossover[idx, 8] += random_value1[5]
        if 0 < offspring_crossover[idx, 9] + random_value1[6] < limits[9]:
            offspring_crossover[idx, 9] += random_value1[6]
        if 0 < offspring_crossover[idx, 10] + random_value2[3] < limits[10]:
            offspring_crossover[idx, 10] += random_value2[3]
            
# Set the working directory
os.chdir('C:/FactSage/')
cwd = os.getcwd()
cwd

# Read the Excel file
df = pd.read_excel('Mg-GA-Added.xlsx')
df = df.sample(frac=1)  # Shuffle the data
df = df.drop('Mg', axis=1)  # Drop the 'Mg' column

# Perform data transformations
y2 = pd.get_dummies(df.iloc[:, 11], prefix='Heat Treatment')
res = pd.concat([df, y2], axis=1)
res = res.drop(['Heat Treatment', 'Heat Treatment_F'], axis=1)
first_col = res.pop("Heat Treatment_T6")
res.insert(11, "Heat Treatment_T6", first_col)

# Describe the data
res.describe().T
res.head()

# Define column names
columns = res.columns

# Define the lists of chemical compounds
Alkline = ['Sr2Mg17(s)', 'Sr5Si3(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)', 'Mg2Ca(s)', 'Al2Ca(s)', 'CaMgSi(s)', 'Mn2CaAl10(s)', 'Ca2Mg55Zn43(s)']
Transition = ['Zr(s)', 'Al3Zr(s)', 'Mn2Zr(s)', 'ZnZr(s)', 'Zn2Zr(s)', 'Y(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)', 'Y6Mn23(s)', 'YZnMg12(s)', 'Mn(s)', 'Al4Mn(s)', 'Al11Mn4(s)', 'Mn3Si(s)', 'Mn5Si3(s)', 'Mn2CaAl10(s)', 'Al7CuMn2(s)', 'Y6Mn23(s)', 'Mn2Zr(s)', 'Mg12Zn13(s)', 'Ca2Mg55Zn43(s)', 'SrZn5(s)', 'Sr2Zn43Mg55(s)', 'YZnMg12(s)', 'Zn2Zr(s)', 'ZnZr(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)', 'Nd2Zn9Mg5(s)', 'NdZn2Al2(s)', 'GdZnMg12(s)', 'Mg2Cu(s)', 'Al7Cu3Mg6(s)', 'Al5Cu6Mg2(s)', 'Al7CuMn2(s)']
Lanthanides = ['GdMg5(s)', 'GdZnMg12(s)', 'Nd5Mg41(s)', 'Nd3Al11(s)', 'Nd3Zn22(s)', 'NdZn2Mg(s)', 'Nd2Zn9Mg5(s)', 'NdZn2Al2(s)']
Post_transition = ['Al30Mg23(s)', 'Al2Ca(s)', 'Mn2CaAl10(s)', 'Al7Cu3Mg6(s)', 'Al5Cu6Mg2(s)', 'Al7CuMn2(s)', 'AlY(s)', 'Al3Y(s)', 'Al2Y3(s)', 'Al3Zr(s)', 'Nd3Al11(s)', 'NdZn2Al2(s)']
Metalloids = ['Mg2Si(s)', 'CaMgSi(s)', 'Mn3Si(s)', 'Mn5Si3(s)', 'Sr5Si3(s)']

# Create empty lists to store the column indices of each category
index_no_Alk = []
index_no_Trans = []
index_no_Lan = []
index_no_Post = []
index_no_Metal = []

# Iterate over the Alkline compounds and get their column indices
for compound in Alkline:
    index_no_Alk.append(res.columns.get_loc(compound))
    
# Iterate over the Transition compounds and get their column indices
for compound in Transition:
    index_no_Trans.append(res.columns.get_loc(compound))

# Iterate over the Lanthanides compounds and get their column indices
for compound in Lanthanides:
    index_no_Lan.append(res.columns.get_loc(compound))
    
# Iterate over the Post_transition compounds and get their column indices
for compound in Post_transition:
    index_no_Post.append(res.columns.get_loc(compound))
    
# Iterate over the Metalloids compounds and get their column indices
for compound in Metalloids:
    index_no_Metal.append(res.columns.get_loc(compound))

# Create a copy of the 'res' dataframe
result = res.copy()

# Calculate the sum of each category and assign the values to new columns
result['Alkline'] = result[Alkline].sum(axis=1)
result['Transition'] = result[Transition].sum(axis=1)
result['Lanthanides'] = result[Lanthanides].sum(axis=1)
result['Post_transition'] = result[Post_transition].sum(axis=1)
result['Metalloids'] = result[Metalloids].sum(axis=1)

# Calculate the sum of all categories and assign the values to a new column
result['SUM'] = result['Alkline'] + result['Transition'] + result['Lanthanides'] + result['Post_transition'] + result['Metalloids']

# Drop the unnecessary columns from index 12 to 52
result = result.drop(result.iloc[:, 12:52], axis=1)

# Reorder the columns to move 'YS (MPa)' to index 20
first_col = result.pop('YS (MPa)')
result.insert(20, 'YS (MPa)', first_col)

# Reorder the columns to move 'UTS (MPa)' to index 19
first_col = result.pop('UTS (MPa)')
result.insert(19, 'UTS (MPa)', first_col)

# Reorder the columns to move 'El%' to index 18
first_col = result.pop('El%')
result.insert(18, 'El%', first_col)

# Create a new dataframe 'result_Y' by dropping rows with missing values in the 'YS (MPa)' column
result_Y = result.dropna(subset=['YS (MPa)'])

# Extract the feature matrix X_Y and the target variable Y_Y from the 'result_Y' dataframe
X_Y = result_Y.iloc[:, :18]
Y_Y = result_Y.iloc[:, 20]
print(np.shape(X_Y), np.shape(Y_Y))


# Create a RandomForestRegressor with specified random state and number of estimators
Rg2 = RandomForestRegressor(random_state=33, n_estimators=8)
Rg2.fit(X_Y, Y_Y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_Y, Y_Y, test_size=0.2, shuffle=True, random_state=3)

# Create a new RandomForestRegressor
Rg = RandomForestRegressor()
Rg.fit(x_train, y_train)
y_pred = Rg.predict(x_test)

# Calculate the mean absolute error, mean squared error, and R2 score
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

# Import the mean function from the statistics module
from statistics import mean

# Perform cross-validation and calculate the mean accuracy
scores = cross_val_score(Rg2, X_Y, Y_Y, cv=10)
print("The accuracy of the model is %f" % (mean(scores)))

# Define a fitness function
def fitness_func(x):
    return Rg2.predict(x)

# Create the initial population
pop_size = 20
population_list = []
for _ in range(pop_size):
    population_list.append([random.uniform(0, 0.05), random.uniform(0, 0.05), random.uniform(0, 0.2), random.uniform(0, 0.05), 0, 0, random.uniform(0, 0.05), 0, 0, 0, 0, 0])
population_list = np.array(population_list)
print(np.shape(population_list))

# Write the population list to an Excel file
with xlsxwriter.Workbook('parham.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    for row_num, data in enumerate(population_list):
        worksheet.write_row(row_num, 0, data)
os.startfile("C:\FactSage\Run_Equilib.bat")

time.sleep(50)

# Read the resulting data from the Excel file
new = pd.read_excel('parham.xlsx', header=None)

# Calculate the sums for each category
new['Alkline'] = new[index_no_Alk].sum(axis=1)
new['Transition'] = new[index_no_Trans].sum(axis=1)
new['Lanthanides'] = new[index_no_Lan].sum(axis=1)
new['Post_transition'] = new[index_no_Post].sum(axis=1)
new['Metalloids'] = new[index_no_Metal].sum(axis=1)

# Calculate the sum of all categories
new['SUM'] = new['Alkline'] + new['Transition'] + new['Lanthanides'] + new['Post_transition'] + new['Metalloids']

# Drop unnecessary columns
new = new.drop(new.columns[12:52], axis=1)

# Convert the data to a NumPy array
new = np.array(new)

# Predict the fitness using the RandomForestRegressor
p = Rg2.predict(new)

final_result = []
final_pop = []
num_parents_mating = 5
num_generations = 250
best_result = []

for generation in range(num_generations):
    print("Generation: ", generation)
    
    # Measure the fitness of each chromosome in the population
    fitness = fitness_func(new)
    
    # Select the best parents in the population for mating
    parents = select_mating_pool(population_list, fitness, num_parents_mating)
    
    # Generate the next generation using crossover
    offspring_crossover = crossover(parents, offspring_size=(pop_size - parents.shape[0], 12))
    
    # Add some variations to the offspring using mutation
    offspring_mutation = mutation(offspring_crossover)
    
    # Create the new population based on the parents and offspring
    population_list[0:parents.shape[0], :] = parents
    population_list[parents.shape[0]:, :] = offspring_mutation
    
    # Write the population to an Excel file
    with xlsxwriter.Workbook('parham.xlsx') as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(population_list):
            worksheet.write_row(row_num, 0, data)
    os.startfile("C:\FactSage\Run_Equilib.bat")
    time.sleep(100)
    
    # Read the resulting data from the Excel file
    new = pd.read_excel('parham.xlsx', header=None)
    
    # Calculate the sums for each category
    new['Alkline'] = new[index_no_Alk].sum(axis=1)
    new['Transition'] = new[index_no_Trans].sum(axis=1)
    new['Lanthanides'] = new[index_no_Lan].sum(axis=1)
    new['Post_transition'] = new[index_no_Post].sum(axis=1)
    new['Metalloids'] = new[index_no_Metal].sum(axis=1)
    
    # Calculate the sum of all categories
    new['SUM'] = new['Alkline'] + new['Transition'] + new['Lanthanides'] + new['Post_transition'] + new['Metalloids']
    
    # Drop unnecessary columns
    new = new.drop(new.columns[12:52], axis=1)
    
    # Convert the data to a NumPy array
    new = np.array(new)
    
    final_pop.append(new)
    final_result.append(fitness_func(new))
    
    # The best result in the current iteration
    print("Best result: ", np.max(fitness_func(new)))
    best_result.append(np.max(fitness_func(new)))

plt.plot(np.arange(250), best_result)

f = np.reshape(final_pop, (5000, 18))
p = np.reshape(final_result, (5000, 1))

fp = pd.DataFrame(f[-20:, :])
pp = pd.DataFrame(p[-20:])

fppp = pd.concat([fp, pp], axis=1)
fppp.head(20)

avg = []
for i in range(0, 5000, 20):
    avg.append(np.mean(p[i:i+20]))

avg_pop = []
for i in range(0, 5000, 20):
    avg_pop.append(np.mean(f[i:i+20, :], axis=0))

plt.plot(np.arange(250), avg)

plt.plot(p, f[:, 5])

fitness = fitness_func(new)
best_match_idx = np.where(fitness == np.max(fitness))
print("Best solution: ", population_list[best_match_idx, :])
print("Best solution fitness: ", fitness[best_match_idx])

m = population_list[best_match_idx, :]
m = np.reshape(m, (1, 12))
res = pd.DataFrame(columns=columns[:12], data=m)
res.head()

n = pd.DataFrame(final_pop)
