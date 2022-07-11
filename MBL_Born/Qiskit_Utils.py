import qiskit
import math
# Takes in a circuit and append an efficient XXYYZZ unitary to it: N(a, b, c) = exp[i(a XX + b YY + c ZZ)]
def N_a_b_c(circuit, a, b, c, q1, q2): 
	theta = +(2*c-math.pi/2)
	phi = +(math.pi/2-2*a)
	lamb = +(2*b-math.pi/2)

	circuit.rz(math.pi/2, q2)
	circuit.cnot(q2, q1)
	circuit.rz(theta, q1)
	circuit.ry(phi, q2)
	circuit.cnot(q1, q2)
	circuit.ry(lamb, q2)
	circuit.cnot(q2, q1)
	circuit.rz(-math.pi/2, q1)

	Ph4(circuit, math.pi/4, qubits=[q1, q2])
	return

def Create_XX_YY_ZZ_Instruction(name, n_qubits, coef, Jx, Jy, Jz, connections):# coef can be either delta_t or delta_t/2
	circuit = qiskit.QuantumCircuit(n_qubits, name=name)
	a = -Jx*coef
	b = -Jy*coef
	c = -Jz*coef
	for con in connections: # Assuming connections has been split into odd and even indices in the case of symmetrized trotterization
		N_a_b_c(circuit, a, b, c, con[0], con[1])
	#circuit.draw() # for debugging
	custom_gate = circuit.to_instruction()
	return custom_gate

# Create the symmetirzed trotterization for the xxz model, coef is the coefficient to scale teh J's
def create_B_C(n_qubits, delta_t, Jx, Jy, Jz, connections):
        even_cs = [] # to store even index (for the first index) connection tuples such as (0,1) (2,3)
        odd_cs = [] # .....such as (1, 2), (5, 8)
        for c in connections:
            if (c[0]+1)%2==0:
                even_cs.append(c)
            else:
                odd_cs.append(c)
        B_gate = Create_XX_YY_ZZ_Instruction("B", n_qubits, delta_t/2, Jx, Jy, Jz, connections=even_cs)
        C_gate = Create_XX_YY_ZZ_Instruction("C", n_qubits, delta_t, Jx, Jx, Jz, connections=odd_cs)
        return B_gate, C_gate

def create_B_B(n_qubits, delta_t, Jx, Jy, Jz, connections):
    even_cs = [] # to store even index (for the first index) connection tuples such as (0,1) (2,3)
    odd_cs = [] # .....such as (1, 2), (5, 8)
    for c in connections:
        if (c[0]+1)%2==0:
            even_cs.append(c)
        else:
            odd_cs.append(c)
    B_even_gate = Create_XX_YY_ZZ_Instruction("B_e", n_qubits, delta_t, Jx, Jy, Jz, connections=even_cs)
    B_odd_gate = Create_XX_YY_ZZ_Instruction("B_o", n_qubits, delta_t, Jx, Jy, Jz, connections=odd_cs)
    return B_even_gate, B_odd_gate



def Neel_state(circuit):
	n_qubits = circuit.num_qubits
	for i in range(n_qubits):
		if i%2 != 0: # for odd numbered indices we do a bit-flip 
			circuit.x(i)
	return

def Wall_state(circuit):
	n_qubits = circuit.num_qubits
	for i in range(int(n_qubits/2)):
		circuit.x(i) # Flip the 1st half of qubits
	return

# Global phase shift gate on U(4) gates
def Ph4(quantum_circuit, theta, qubits):
    quantum_circuit.u1(theta, qubits[0])
    quantum_circuit.u1(theta, qubits[1])
    quantum_circuit.x(qubits[0])
    quantum_circuit.x(qubits[1])
    quantum_circuit.u1(theta, qubits[0])
    quantum_circuit.u1(theta, qubits[1])
    quantum_circuit.x(qubits[0])
    quantum_circuit.x(qubits[1])


