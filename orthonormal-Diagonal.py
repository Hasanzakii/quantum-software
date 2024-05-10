import numpy as np


pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])

eigenvalues_x, eigenvectors_x = np.linalg.eig(pauli_x)
eigenvalues_y, eigenvectors_y = np.linalg.eig(pauli_y)
eigenvalues_z, eigenvectors_z = np.linalg.eig(pauli_z)

print("Pauli Matrix σx:")
print("Eigenvectors:")
print(eigenvectors_x)
print("Eigenvalues:")
print(eigenvalues_x)

print("\nPauli Matrix σy:")
print("Eigenvectors:")
print(eigenvectors_y)
print("Eigenvalues:")
print(eigenvalues_y)

print("\nPauli Matrix σz:")
print("Eigenvectors:")
print(eigenvectors_z)
print("Eigenvalues:")
print(eigenvalues_z)

decomposition_x = np.outer(
    eigenvectors_x[:, 0], np.conj(eigenvectors_x[:, 0])
) - np.outer(eigenvectors_x[:, 1], np.conj(eigenvectors_x[:, 1]))
decomposition_y = -1j * np.outer(
    eigenvectors_y[:, 0], np.conj(eigenvectors_y[:, 0])
) + 1j * np.outer(eigenvectors_y[:, 1], np.conj(eigenvectors_y[:, 1]))
decomposition_z = np.outer(
    eigenvectors_z[:, 0], np.conj(eigenvectors_z[:, 0])
) - np.outer(eigenvectors_z[:, 1], np.conj(eigenvectors_z[:, 1]))

print("\nOrthonormal Decompositions:")
print("Pauli Matrix σx:")
print(decomposition_x)
print("Pauli Matrix σy:")
print(decomposition_y)
print("Pauli Matrix σz:")
print(decomposition_z)
