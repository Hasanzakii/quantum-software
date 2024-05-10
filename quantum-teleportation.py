
import random
import matplotlib.pyplot as plt
import numpy as np
import cirq

def make_quantum_teleportation_circuit(gate):
    """Returns a circuit for quantum teleportation.

    This circuit 'teleports' a random qubit state prepared by
    the input gate from Alice to Bob.
    """
    circuit = cirq.Circuit()

    
    msg = cirq.NamedQubit("Message")
    alice = cirq.NamedQubit("Alice")
    bob = cirq.NamedQubit("Bob")

    
    circuit.append(gate(msg))

    
    circuit.append([cirq.H(alice), cirq.CNOT(alice, bob)])

    
    circuit.append([cirq.CNOT(msg, alice), cirq.H(msg), cirq.measure(msg, alice)])

    
    
    circuit.append([cirq.CNOT(alice, bob), cirq.CZ(msg, bob)])

    return circuit

"""Visualize the teleportation circuit."""

gate = cirq.X ** 0.25


circuit = make_quantum_teleportation_circuit(gate)
print("Teleportation circuit:\n")
print(circuit)

"""Display the Bloch vector of the message qubit."""
message = cirq.Circuit(gate.on(cirq.NamedQubit("Message"))).final_state_vector()
message_bloch_vector = cirq.bloch_vector_from_state_vector(message, index=0)
print("Bloch vector of message qubit:")
print(np.round(message_bloch_vector, 3))

"""Simulate the teleportation circuit and get the final state of Bob's qubit."""

sim = cirq.Simulator()


result = sim.simulate(circuit)


bobs_bloch_vector = cirq.bloch_vector_from_state_vector(result.final_state_vector, index=1)
print("Bloch vector of Bob's qubit:")
print(np.round(bobs_bloch_vector, 3))


np.testing.assert_allclose(bobs_bloch_vector, message_bloch_vector, atol=1e-6)

def make_qft(qubits):
    """Generator for the QFT on a list of qubits."""
    qreg = list(qubits)
    while len(qreg) > 0:
        q_head = qreg.pop(0)
        yield cirq.H(q_head)
        for i, qubit in enumerate(qreg):
            yield (cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head)

"""Visually check the QFT circuit."""
qubits = cirq.LineQubit.range(4)
qft = cirq.Circuit(make_qft(qubits))
print(qft)

"""Use the built-in QFT in Cirq."""
qft_operation = cirq.qft(*qubits, without_reverse=True)
qft_cirq = cirq.Circuit(qft_operation)
print(qft_cirq)

"""Check equality of the 'manual' and 'built-in' QFTs."""
np.testing.assert_allclose(cirq.unitary(qft), cirq.unitary(qft_cirq))

def make_qft_inverse(qubits):
    """Generator for the inverse QFT on a list of qubits.

    For four qubits, the answer is:

                       ┌────────┐   ┌──────────────┐   ┌────────┐
    0: ───H───@─────────@────────────@───────────────────────────────────────────×───
              │         │            │                                           │
    1: ───────@^-0.5────┼──────H─────┼───────@──────────@────────────────────×───┼───
                        │            │       │          │                    │   │
    2: ─────────────────@^-0.25──────┼───────@^-0.5─────┼──────H────@────────×───┼───
                                     │                  │           │            │
    3: ──────────────────────────────@^(-1/8)───────────@^-0.25─────@^-0.5───H───×───
                       └────────┘   └──────────────┘   └────────┘
    """
    

def make_qft_inverse(qubits):
    qreg = list(qubits)[::-1]
    while len(qreg) > 0:
        q_head = qreg.pop(0)
        yield cirq.H(q_head)
        for i, qubit in enumerate(qreg):
            yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)

"""Visually check the inverse QFT circuit."""
qubits = cirq.LineQubit.range(4)
iqft = cirq.Circuit(make_qft_inverse(qubits))
print(iqft)

"""Use the built-in inverse QFT in Cirq."""
iqft_operation = cirq.qft(*qubits, inverse=True, without_reverse=True)
iqft_cirq = cirq.Circuit(iqft_operation)
print(iqft_cirq)

"""Visually check the inverse QFT circuit."""
qubits = cirq.LineQubit.range(4)
iqft = cirq.Circuit(make_qft_inverse(qubits))
print(iqft)


"""Use the built-in inverse QFT in Cirq."""
iqft_operation = cirq.qft(*qubits, inverse=True, without_reverse=True)
iqft_cirq = cirq.Circuit(iqft_operation)
print(iqft_cirq)


"""Check equality of the 'manual' and 'built-in' inverse QFTs."""
np.testing.assert_allclose(cirq.unitary(iqft), cirq.unitary(iqft_cirq))

"""Set up the unitary and number of bits to use in phase estimation."""


theta = 0.234


U = cirq.Z ** (2 * theta)


n_bits = 3

"""Build the first part of the circuit for phase estimation."""

qubits = cirq.LineQubit.range(n_bits)
u_bit = cirq.NamedQubit('u')


phase_estimator = cirq.Circuit(cirq.H.on_each(*qubits))

for i, bit in enumerate(qubits):
    phase_estimator.append(cirq.ControlledGate(U).on(bit, u_bit) ** (2 ** (n_bits - i - 1)))

print(phase_estimator)


"""Build the last part of the circuit (inverse QFT) for phase estimation."""

phase_estimator.append(make_qft_inverse(qubits[::-1]))


phase_estimator.append(cirq.measure(*qubits, key='m'))
print(phase_estimator)



"""Set the input state of the eigenvalue register."""

phase_estimator.insert(0, cirq.X(u_bit))

print(phase_estimator)


"""Simulate the circuit and convert from measured bit values to estimated θ values."""

sim = cirq.Simulator()
result = sim.run(phase_estimator, repetitions=10)


theta_estimates = np.sum(2 ** np.arange(n_bits) * result.measurements['m'], axis=1) / 2**n_bits
print(theta_estimates)



def phase_estimation(theta, n_bits, n_reps=10, prepare_eigenstate_gate=cirq.X):
    """Runs the phase estimate algorithm for unitary U=Z^{2θ} with n_bits qubits."""
    
    qubits = cirq.LineQubit.range(n_bits)
    u_bit = cirq.NamedQubit('u')

    
    U = cirq.Z ** (2 * theta)

    
    

    
    phase_estimator.insert(0, prepare_eigenstate_gate.on(u_bit))

    
    

    return theta_estimates


def phase_estimation(theta, n_bits, n_reps=10, prepare_eigenstate_gate=cirq.X):
    
    qubits = cirq.LineQubit.range(n_bits)
    u_bit = cirq.NamedQubit('u')

    
    U = cirq.Z ** (2 * theta)

    
    phase_estimator = cirq.Circuit(cirq.H.on_each(*qubits))

    
    for i, bit in enumerate(qubits):
        phase_estimator.append(cirq.ControlledGate(U).on(bit, u_bit) ** (2 ** (n_bits - 1 - i)))

    
    phase_estimator.append(make_qft_inverse(qubits[::-1]))

    
    phase_estimator.append(cirq.measure(*qubits, key='m'))

    
    phase_estimator.insert(0, prepare_eigenstate_gate.on(u_bit))

    
    sim = cirq.Simulator()
    result = sim.run(phase_estimator, repetitions=n_reps)

    
    theta_estimates = np.sum(2**np.arange(n_bits)*result.measurements['m'], axis=1)/2**n_bits

    return theta_estimates


"""Analyze convergence vs n_bits."""

theta = 0.123456

max_nvals = 16
nvals = np.arange(1, max_nvals, step=1)


estimates = []
for n in nvals:
    estimate = phase_estimation(theta=theta, n_bits=n, n_reps=1)[0]
    estimates.append(estimate)


"""Plot the results."""
plt.style.use("seaborn-v0_8-whitegrid")

plt.plot(nvals, estimates, "--o", label="Phase estimation")
plt.axhline(theta, label="True value", color="black")

plt.legend()
plt.xlabel("Number of bits")
plt.ylabel(r"$\theta$");


"""Run phase estimation without starting in an eigenstate."""

theta = 0.123456


n = 4


res = phase_estimation(theta=theta, n_bits=n, n_reps=10, prepare_eigenstate_gate=cirq.H)
print(res)


"""Get qubits to use in the circuit for Grover's algorithm."""

nqubits = 2


qubits = cirq.LineQubit.range(nqubits)
ancilla = cirq.NamedQubit("Ancilla")


def make_oracle(qubits, ancilla, xprime):
    """Implements the function {f(x) = 1 if x == x', f(x) = 0 if x != x'}."""
    
    

    
    yield (cirq.X(q) for (q, bit) in zip(qubits, xprime) if not bit)

    
    yield (cirq.TOFFOLI(qubits[0], qubits[1], ancilla))

    
    yield (cirq.X(q) for (q, bit) in zip(qubits, xprime) if not bit)


def grover_iteration(qubits, ancilla, oracle):
    """Performs one round of the Grover iteration."""

    circuit = cirq.Circuit()
    
    circuit.append(cirq.H.on_each(*qubits))

    
    circuit.append([cirq.X(ancilla), cirq.H(ancilla)])

    
    circuit.append(oracle)

    
    circuit.append(cirq.H.on_each(*qubits))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.H.on(qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.H.on(qubits[1]))
    circuit.append(cirq.X.on_each(*qubits))
    circuit.append(cirq.H.on_each(*qubits))

    
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit    

"""Select a 'marked' bitstring x' at random."""
xprime = [random.randint(0, 1) for _ in range(nqubits)]
print(f"Marked bitstring: {xprime}")

"""Create the circuit for Grover's algorithm."""

oracle = make_oracle(qubits, ancilla, xprime)


circuit = grover_iteration(qubits, ancilla, oracle)
print("Circuit for Grover's algorithm:")
print(circuit)


"""Simulate the circuit for Grover's algorithm and check the output."""

def bitstring(bits):
    return "".join(str(int(b)) for b in bits)


simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=10)


frequencies = result.histogram(key="result", fold_func=bitstring)
print('Sampled results:\n{}'.format(frequencies))


most_common_bitstring = frequencies.most_common(1)[0][0]
print("\nMost common bitstring: {}".format(most_common_bitstring))
print("Found a match? {}".format(most_common_bitstring == bitstring(xprime)))


