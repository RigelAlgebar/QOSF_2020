# Mathematical imports
from scipy.optimize import minimize
import numpy as np
from math import pi

# Quiskit imports
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

# Import this to see the progress of the simulations
from tqdm import trange



################################################################################
################################################################################
################################################################################



# Define the backend
backend = BasicAer.get_backend('statevector_simulator')


# Here we set the seed for the random number generator (for reproducibility)
np.random.seed(101)


# This is the random "PHI" state we optimize against with 
phi = 2*pi*np.random.random(16) + 2*pi*np.random.random(16) * 1j
phi = phi/np.linalg.norm(phi)



################################################################################
################################################################################
################################################################################



class oddBlock:
    
    '''
    The oddBlock object adds a series of four rotation gates all of them in
    the same axis.

    Args:
    	q (Quantum Register): Specify the register on which we act upon.
    	qc (Quantum Circuit): Specifu the circuit were the gates are added.
        vector_angles: 1-D Vector containing all of the angles for the gates.

    Methods:
    	Each method specifies the axis for the rotation gates and returns the
    	input circuit modified in place. 
    '''

    @staticmethod
    def addBlock_xaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.rx(angles[0], q[0])
        qc.rx(angles[1], q[1])
        qc.rx(angles[2], q[2])
        qc.rx(angles[3], q[3])
        return qc
    
    
    @staticmethod
    def addBlock_yaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.ry(angles[0], q[0])
        qc.ry(angles[1], q[1])
        qc.ry(angles[2], q[2])
        qc.ry(angles[3], q[3])
        return qc
    
    
    @staticmethod
    def addBlock_zaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.rz(angles[0], q[0])
        qc.rz(angles[1], q[1])
        qc.rz(angles[2], q[2])
        qc.rz(angles[3], q[3])
        return qc



################################################################################
################################################################################
################################################################################



class evenBlock:
    
    '''
    The evenBlock object adds a series of four rotation gates follow by four
    control raotation gates, all of them acting on the same axis.

    Args:
    	q (Quantum Register): Specify the register on which we act upon.
    	qc (Quantum Circuit): Specify the circuit were the gates are added.
        vector_angles: 1-D Vector containing all of the angles for the gates.

    Methods:
    	Each method specifies the axis for the rotation gates and returns the
    	input circuit modified in place. 
    '''

    @staticmethod
    def addBlock_xaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.rx(angles[0], q[0])
        qc.rx(angles[1], q[1])
        qc.rx(angles[2], q[2])
        qc.rx(angles[3], q[3])
        qc.cx(q[0], q[1])
        qc.cx(q[0], q[2])
        qc.cx(q[0], q[3])
        qc.cx(q[1], q[2])
        qc.cx(q[1], q[3])
        qc.cx(q[2], q[3])
        return qc
    
    
    @staticmethod
    def addBlock_yaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.ry(angles[0], q[0])
        qc.ry(angles[1], q[1])
        qc.ry(angles[2], q[2])
        qc.ry(angles[3], q[3])
        qc.cy(q[0], q[1])
        qc.cy(q[0], q[2])
        qc.cy(q[0], q[3])
        qc.cy(q[1], q[2])
        qc.cy(q[1], q[3])
        qc.cy(q[2], q[3])
        return qc
        
    
    @staticmethod
    def addBlock_zaxis(q, qc, vector_angles):
        angles = vector_angles
        qc.rz(angles[0], q[0])
        qc.rz(angles[1], q[1])
        qc.rz(angles[2], q[2])
        qc.rz(angles[3], q[3])
        qc.cz(q[0], q[1])
        qc.cz(q[0], q[2])
        qc.cz(q[0], q[3])
        qc.cz(q[1], q[2])
        qc.cz(q[1], q[3])
        qc.cz(q[2], q[3])
        return qc



################################################################################
################################################################################
################################################################################
    


class simulation:

	'''
	The simulation object defines a fully initialize circuit to execute.

    Args:
    	q (Quantum Register): Specify the register on which we act upon.
    	qc (Quantum Circuit): Specify the circuit were the gates are added.
        vector_odd_angles: 2-D Vector containing all of the angles for the 
        odd blocks.
        vector_even_angles: 2-D Vector containing all of the angles for the 
        even blocks.
        !!!
        caseNum: This parameter is supposed to help us to specify a simulation
        case; however, when the method "caseToRun" is invoked, the the whole
        simuation presents an unexpected beheavior. I decided to not to depre_
        cate this parameter for the moment as it does not interfere with the
        workflow of the Notebooks. WORK IN PROGRESS.
		!!!

    Methods:
    	Each method represents a study case, being each case a pair of axis on
    	each of the blocks. 
    	The caseToRun method is gibing rise to weird beheavior of the simulation
    	and hence is currently not supported. This method contains a dictionary
    	that shoud call the N-build method vigen an N input from the user.
    '''

	def __init__(self, q, qc, layers, vector_oddAngles, vector_evenAngles, caseNum):
		self.q = QuantumRegister(4)
		self.qc = QuantumCircuit(self.q)
		self.layers = layers
		self.angles_odd = vector_oddAngles
		self.angles_even = vector_evenAngles
		self.caseNum = caseNum
        
        
	def build_case1(self):
		'''
		Odd block  axis: X
		Even block axis: X
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_xaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_xaxis(self.q, self.qc, self.angles_even[i])
        
		return self.qc
		

	def build_case2(self):
		'''
		Odd block  axis: X
		Even block axis: Y
   		'''
		for i in range(self.layers):
		    self.qc = oddBlock.addBlock_xaxis(self.q, self.qc, self.angles_odd[i])
		    self.qc = evenBlock.addBlock_yaxis(self.q, self.qc, self.angles_even[i])
        
		return self.qc
	
	
	def build_case3(self):
		'''
		Odd block  axis: X
		Even block axis: Z
		'''
		for i in range(self.layers):
		    self.qc = oddBlock.addBlock_xaxis(self.q, self.qc, self.angles_odd[i])
		    self.qc = evenBlock.addBlock_zaxis(self.q, self.qc, self.angles_even[i])
		
		return self.qc
    
    
	def build_case4(self):
		'''
		Odd block  axis: Y
		Even block axis: X
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_yaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_xaxis(self.q, self.qc, self.angles_even[i])

		return self.qc
    
    
	def build_case5(self):
		'''
		Odd block  axis: Y
		Even block axis: Y
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_yaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_yaxis(self.q, self.qc, self.angles_even[i])
		
		return self.qc
    
    
	def build_case6(self):
		'''
		Odd block  axis: Y
		Even block axis: Z
		'''
		for i in range(self.layers):
		    self.qc = oddBlock.addBlock_yaxis(self.q, self.qc, self.angles_odd[i])
		    self.qc = evenBlock.addBlock_zaxis(self.q, self.qc, self.angles_even[i])
		
		return self.qc
    
    
	def build_case7(self):
		'''
		Odd block  axis: Z
		Even block axis: X
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_zaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_xaxis(self.q, self.qc, self.angles_even[i])

		return self.qc
    
    
	def build_case8(self):
		'''
		Odd block  axis: Z
		Even block axis: Y
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_zaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_yaxis(self.q, self.qc, self.angles_even[i])
		
		return self.qc

    
	def build_case9(self):
		'''
		Odd block  axis: Z
		Even block axis: Z
		'''
		for i in range(self.layers):
			self.qc = oddBlock.addBlock_zaxis(self.q, self.qc, self.angles_odd[i])
			self.qc = evenBlock.addBlock_zaxis(self.q, self.qc, self.angles_even[i])

		return self.qc
    
    
	# Gives rise to an unexpected beheavior! Therefore, temporarely out of 
	# service.
	'''
	def caseToRun(self):
		cases = {1: self.build_case1(), 2: self.build_case2(), 3: self.build_case3(), 
		4: self.build_case4(), 5: self.build_case5(), 6: self.build_case6(), 
		7: self.build_case7(), 8: self.build_case8(), 9: self.build_case9()}

	return cases[self.caseNum]
	'''



################################################################################
################################################################################
################################################################################



'''
The following functions are a way around for the automatization of the simulation
cases. To implement a proper automatization we need to:
	 i) Fix the "caseToRun" method of the simulation class.
	ii) Find the right way for the wraping of the arguments of minimize function
	of scipy. 

I have limited this document to the first three cases for the moment, as it seems
like the other ones behave qualitatively similar to ones define below. 
'''

def objective_case1(angles):

    q_trial    = QuantumRegister(4)
    qc_trial   = QuantumCircuit(q_trial)


    # The simulation class receives 2-D vectors of 1-D sub-vectors each with 
    # the four angles needed for the rotation gates.
    imp_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(0, len(angles)-4, 8)]
    par_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(4, len(angles), 8)]
    layers     = int(len(angles)/8)
    

    # Here we define a trial simulation to produce the trial state which we use
    # to define the function (norm) that we want to minimize.
    sim_trial   = simulation(q_trial, qc_trial, layers, imp_angles, par_angles, 1).build_case1()
    state_trial = execute(sim_trial, backend).result().get_statevector()
    
    
    # Return the function to minimize, in this case the norm of the difference
    # of or trial state an the reference random state phi.
    return np.linalg.norm(state_trial - phi)



def objective_case2(angles):

    q_trial    = QuantumRegister(4)
    qc_trial   = QuantumCircuit(q_trial)


    # The simulation class receives 2-D vectors of 1-D sub-vectors each with 
    # the four angles needed for the rotation gates.
    imp_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(0, len(angles)-4, 8)]
    par_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(4, len(angles), 8)]
    layers     = int(len(angles)/8)
    

    # Here we define a trial simulation to produce the trial state which we use
    # to define the function (norm) that we want to minimize.
    sim_trial   = simulation(q_trial, qc_trial, layers, imp_angles, par_angles, 2).build_case2()
    state_trial = execute(sim_trial, backend).result().get_statevector()
    
    
    # Return the function to minimize, in this case the norm of the difference
    # of or trial state an the reference random state phi.
    return np.linalg.norm(state_trial - phi)



def objective_case3(angles):

    q_trial    = QuantumRegister(4)
    qc_trial   = QuantumCircuit(q_trial)


    # The simulation class receives 2-D vectors of 1-D sub-vectors each with 
    # the four angles needed for the rotation gates.
    imp_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(0, len(angles)-4, 8)]
    par_angles = [[angles[i], angles[i+1], angles[i+2], angles[i+3]] for i in range(4, len(angles), 8)]
    layers     = int(len(angles)/8)
    

    # Here we define a trial simulation to produce the trial state which we use
    # to define the function (norm) that we want to minimize.
    sim_trial   = simulation(q_trial, qc_trial, layers, imp_angles, par_angles, 3).build_case3()
    state_trial = execute(sim_trial, backend).result().get_statevector()
    
    
    # Return the function to minimize, in this case the norm of the difference
    # of or trial state an the reference random state phi.
    return np.linalg.norm(state_trial - phi)



################################################################################
################################################################################
################################################################################



'''
Same as before, the following functions are a way around for the automatization 
of the simulation cases.
'''

def optimization_case1(layer, odd_block_angles, even_block_angles):
    
    # The "minimize" function from scipy only accepts 1-D arrays as argument,
    # so in the following lines e take the odd and even vectors, stack them
    # together (horizontally) and flatten the resultant array.
    angles = np.hstack([odd_block_angles[:layer+1], even_block_angles[:layer+1]])
    angles = angles.flatten()
    

    # Boundaries (limits) for each variational paramater.
    bnds   = tuple((0, 2.0*pi) for i in range(len(angles)))

    # Minimize the specific case using the L-BFGS-B' method.
    result = minimize(objective_case1, angles, method='L-BFGS-B', bounds=bnds)
    

    # Here we re-order the 1-D array from the minimization result as the new
    # vectors for the odd and even angles in the right format for the sumulation
    new_odd_angles  = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(0, len(angles)-4, 8)]
    new_even_angles = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(4, len(angles), 8)]
    

    # Return the optimized vectors
    return [new_odd_angles, new_even_angles]




def optimization_case2(layer, odd_block_angles, even_block_angles):
    
    # The "minimize" function from scipy only accepts 1-D arrays as argument,
    # so in the following lines e take the odd and even vectors, stack them
    # together (horizontally) and flatten the resultant array.
    angles = np.hstack([odd_block_angles[:layer+1], even_block_angles[:layer+1]])
    angles = angles.flatten()
    
    # Boundaries (limits) for each variational paramater.
    bnds   = tuple((0, 2.0*pi) for i in range(len(angles)))

    # Minimize the specific case using the L-BFGS-B' method.
    result = minimize(objective_case2, angles, method='L-BFGS-B', bounds=bnds)
    

    # Here we re-order the 1-D array from the minimization result as the new
    # vectors for the odd and even angles in the right format for the sumulation
    new_odd_angles  = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(0, len(angles)-4, 8)]
    new_even_angles = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(4, len(angles), 8)]
    

    # Return the optimized vectors
    return [new_odd_angles, new_even_angles]



def optimization_case3(layer, odd_block_angles, even_block_angles):#OJO AQUI
    
    # The "minimize" function from scipy only accepts 1-D arrays as argument,
    # so in the following lines e take the odd and even vectors, stack them
    # together (horizontally) and flatten the resultant array.
    angles = np.hstack([odd_block_angles[:layer+1], even_block_angles[:layer+1]])
    angles = angles.flatten()
    
    # Minimize the specific case using the L-BFGS-B' method.
    bnds   = tuple((0, 2.0*pi) for i in range(len(angles)))

    # Minimize the specific case using the L-BFGS-B' method.
    result = minimize(objective_case3, angles, method='L-BFGS-B', bounds=bnds)
    

    # Here we re-order the 1-D array from the minimization result as the new
    # vectors for the odd and even angles in the right format for the sumulation
    new_odd_angles  = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(0, len(angles)-4, 8)]
    new_even_angles = [[result.x[i], result.x[i+1], result.x[i+2], result.x[i+3]] for i in range(4, len(angles), 8)]
    

    # Return the optimized vectors
    return [new_odd_angles, new_even_angles]



################################################################################
################################################################################
################################################################################
