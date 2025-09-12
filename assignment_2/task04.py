import numpy as np

A = np.array([[1,2,3], [0,1,4], [5,6,0]])
inverse_of_A = np.linalg.inv(A)

I1 = np.dot(A, inverse_of_A)
I2 = np.dot(inverse_of_A, A)
print("Matrix A: ")
print(A)

print("\nInverse of A: ")
print(inverse_of_A)

print("\nInverse of A: ")
print(I1)

print("\nInverse of A*A: ")
print(I2)

print("\nCheck:")
print("A*Inverse_of_A is identity:", np.allclose(I1, np.eye(3)))
print("Inverse_of_OfA*A is identity:", np.allclose(I2, np.eye(3)))