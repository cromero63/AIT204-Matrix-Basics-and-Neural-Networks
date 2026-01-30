import streamlit as st

st.title("Matrix Basics and Neural Networks")

st.markdown("""
    ### Part 1

    a. How do you represent an $a_{i,j}$ matrix?

    &emsp; An $a_{i,j}$ matrix represents a matrix with $i$ rows and $j$ columns.


    b. Represent a row vector and a column vector.

    &emsp;**Row vector:** a matrix with only one row.

    &emsp;&emsp; $R = [r_1, r_2, ..., r_n]$

    &emsp;**Column vector:** a matrix with only one column.

    &emsp;&emsp; $C = \\begin{bmatrix} c_1 \\\\ c_2 \\\\ \\vdots \\\\ c_m \\end{bmatrix}$

    c. Let $A = \\begin{bmatrix} 1 & -2 & 3 \\\\ 4 & 5 & -6 \\end{bmatrix}$ and $B = \\begin{bmatrix} 3 & 0 & 2 \\\\ -7 & 1 & 8 \\end{bmatrix}$ find:

    1. $A + B$

    &emsp; $\\begin{bmatrix} 4 & -2 & 5 \\\\ -3 & 6 & 2 \\end{bmatrix}$

    2. $3A$

    &emsp; $\\begin{bmatrix} 3 & -6 & 9 \\\\ 12 & 15 & -18 \\end{bmatrix}$

    3. $2A - 3B$

    &emsp; $\\begin{bmatrix} 7 & -4 & 0 \\\\ 29 & 7 & -36 \\end{bmatrix}$

    4. $AB$

        Not defined. AB is not possible, because matrix multiplication requires the number of columns in the first matrix to equal the number of columns in the second matrix. In this case both matrices are 2x3.

    5. $A^T$

    &emsp; $\\begin{bmatrix} 1 & 4 \\\\ -2 & 5 \\\\ 3 & -6 \\end{bmatrix}$

    6. $AI$

    &emsp; $\\begin{bmatrix} 1 & -2 & 3 \\\\ 4 & 5 & -6 \\end{bmatrix}$

    """)
