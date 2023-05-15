# Macro Code for Excel Sheet Manipulation

This repository contains a macro code written in Excel VBA to manipulate an Excel sheet named "Data.xlsx" in order to read and write data. The macro code performs specific operations on the sheet to extract and calculate results based on the provided composition data.

## Overview

The macro code performs the following operations on the "Data.xlsx" sheet:

1. Reading Composition Data:
   - Reads material composition data from cells A2 in the "parham.xlsx" sheet.
   - The composition data is stored in an array variable called `%comp`.

2. Opening an Equilibrium Calculation Program:
   - Opens a program called "P.equi" for performing equilibrium calculations.

3. Looping over Variables:
   - Iterates from 1 to 3 using the variable `%2`.
   - Within each iteration:
     - Reads composition data from cell A%2 in the "Data.xlsx" sheet.
     - Iterates from 1 to 11 using the variable `%1`.
       - Sets reactant mass based on the composition data.
     - Sets reactant 12 mass as 100 (placeholder value).
     - Performs calculations using the equilibrium program.
     - Stores the calculated results in an array variable called `%Results_mass`.
     - Writes the results to the "Data.xlsx" sheet under column M%2.

4. Closing and Saving the Sheet:
   - Closes the "parham.xlsx" sheet and saves the changes.

## Contributing

We welcome contributions to enhance the macro code. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Make your changes and test them.
4. Submit a pull request describing your changes.

## Contact

For questions, issues, or collaboration opportunities, please contact [Parham Valipoorsalimi] at [Parham.valipoorsalimi@mail.mcgill.ca].
