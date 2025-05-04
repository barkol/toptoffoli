from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile 
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

class Shift:
    def __init__(self,numOfAncillas = 3,numOfQbits = 2,ancillaNum = 0):
        assert ancillaNum <= numOfAncillas-1 or numOfAncillas == 0 , "Your ancilla number exceeds given range"
        self.nA = numOfAncillas
        self.nQ = numOfQbits
        self.n = self.nA + self.nQ
        self.k = ancillaNum
        self.U = self.circ()
    
    def circ(self):
        qc = QuantumCircuit(self.nA + self.nQ)
        control_tab = ([self.k] if (self.k >= 0) else []) + [( self.nA + i_ ) for i_ in range(self.nQ)]
        for i in range(2**(self.nQ+1)-1):
            string = Shift.DecToBin(i,self.nQ+1)
            tab_of_diff = [((i_ + self.nA - 1) if (i_ != 0) else self.k) for i_ in range(len(string)) if string[i_] != "1"]
            temp_qc = QuantumCircuit(self.nA + self.nQ)
            for j in range(len(tab_of_diff)):
                if j == 0:
                    temp_qc = temp_qc.compose(self.set_gate(tab_of_diff,control_tab,j))
                else:
                    temp_qc2 = QuantumCircuit(self.nA + self.nQ)
                    temp_qc2 = temp_qc2.compose(self.set_gate(tab_of_diff,control_tab,j))
                    temp_qc2 = temp_qc2.compose(temp_qc)
                    temp_qc = temp_qc2.compose(self.set_gate(tab_of_diff,control_tab,j))
            qc = qc.compose(temp_qc)
        return qc

    def set_gate(self,tab_of_diff,control_tab,j):
        temp_qc = QuantumCircuit(self.nA + self.nQ)
        temp_qc.mcx(list(set(control_tab) - set([tab_of_diff[j]])),tab_of_diff[j])
        for k in range(j+1,len(tab_of_diff)):
            temp_qc.x(tab_of_diff[k])
            temp_qc_x = QuantumCircuit(self.nA + self.nQ)
            temp_qc_x.x(tab_of_diff[k])
            temp_qc = temp_qc_x.compose(temp_qc)
        return temp_qc

    def get_qc(self):
        return self.U
    
    @staticmethod
    def DecToBin(num_2,max):
        def internal(num):
            if num == 0:
                return "0"
            elif num == 1:
                return "1"
            else:
                return internal(num // 2) + str(num % 2) 
        ans = internal(num_2)
        return "0" * (max - len(ans)) + ans
    
class ShiftExperiment_FlowControl:
    def __init__(self,qubits = 2,ancillas = 3,controlling_anc = 2):
        self.qubits = qubits
        self.ancillas = ancillas
        self.controlling_anc = controlling_anc
        self.Part2 = self.getEnding()

    def getEnding(self):
        model = Shift(self.ancillas,self.qubits,self.controlling_anc)
        qc = model.get_qc()
        EndPoints = ClassicalRegister(self.qubits + 1,"c")
        qc.add_register(EndPoints)
        qc.barrier()
        qc.measure(self.controlling_anc, EndPoints[2])  
        qc.measure(self.ancillas, EndPoints[1])
        qc.measure(self.ancillas+1, EndPoints[0])
        return qc
    
    def run(self,initial_state_tab,**kwargs):
        Starting = QuantumCircuit(
            QuantumRegister(self.ancillas,"a"),
            QuantumRegister(self.qubits,"q")
        )
        for i in range(len(initial_state_tab)):
            Starting.initialize(initial_state_tab[i],i)
        qc = Starting.compose(self.Part2)

        sc = kwargs.get("show_circ",False)
        if sc:
            fig = plt.figure(figsize = (10,5))
            ax = fig.add_subplot()
            qc.draw('mpl',ax = ax)
        simulator = AerSimulator()
        qc_comp = transpile(qc,simulator)
        res = simulator.run(qc_comp).result()
        return res.get_counts(qc_comp)
    
class SynchExperiment_FlowControl:
    def __init__(self,ancillas,qubits):
        self.ancillas = ancillas
        self.qubits = qubits
        self.Part2 = self.getEnding()

    def getEnding(self):
        Shifts = [Shift(self.ancillas,self.qubits,self.ancillas - 1 - i) for i in range(self.ancillas)]
        Ts = [self.T_(self.ancillas - 1 - i) for i in range(self.ancillas)]
        qc = QuantumCircuit(self.qubits + self.ancillas)
        for i in range(len(Shifts)):
            qc = qc.compose(Ts[i])
            qc.barrier()
            qc = qc.compose(Shifts[i].get_qc())
            qc.barrier()
            qc = qc.compose(Ts[i])
            qc.barrier()
        
        EndPoints = ClassicalRegister(self.qubits,"c")
        qc.add_register(EndPoints)
        for i in range(self.qubits):
            qc.measure(self.ancillas+i,EndPoints[self.qubits - 1 -i])
        return qc

    def T_(self,a):
        T1 = QuantumCircuit(self.qubits + self.ancillas)
        for i in range(self.qubits - 1):
            T1.x(self.ancillas + i)
        T1.mcx([a] + [self.ancillas + i for i in range(self.qubits -1)],self.ancillas + self.qubits - 1)
        for i in range(self.qubits - 1):
            T1.x(self.ancillas + i)
        return T1    
        
    def run(self,initial_state_tab,**kwargs):
        Starting = QuantumCircuit(
            QuantumRegister(self.ancillas,"a"),
            QuantumRegister(self.qubits,"q")
        )
        for i in range(len(initial_state_tab)):
            Starting.initialize(initial_state_tab[i],i)
        for i in range(self.qubits):
            Starting.h(self.ancillas + i)
        Starting.barrier()
        qc = Starting.compose(self.Part2)

        sc = kwargs.get("show_circ",False)
        if sc:
            fig = plt.figure(figsize = (15,15))
            ax = fig.add_subplot()
            qc.draw('mpl',ax = ax)
        simulator = AerSimulator()
        qc_comp = transpile(qc,simulator)
        res = simulator.run(qc_comp).result()
        return res.get_counts(qc_comp)
    
class SynchExperiment_FlowControl_Entanglement(SynchExperiment_FlowControl):
    def getEnding(self):
        Shifts = [Shift(self.ancillas,self.qubits,self.ancillas - 1 - i) for i in range(self.ancillas)]
        Ts = [self.T_(self.ancillas - 1 - i) for i in range(self.ancillas)]
        qc = QuantumCircuit(self.qubits + self.ancillas)
        for i in range(len(Shifts)):
            qc = qc.compose(Ts[i])
            qc.barrier()
            qc = qc.compose(Shifts[i].get_qc())
            qc.barrier()
            qc = qc.compose(Ts[i])
            qc.barrier()
        qc.add_register(QuantumRegister(1))
        qc.barrier()

        EndPoints = ClassicalRegister(self.qubits+self.ancillas +1,"c")
        qc.add_register(EndPoints)
        for i in range(self.qubits+self.ancillas+1):
            qc.measure(i,EndPoints[self.qubits+self.ancillas-i])
        return qc
    
    def get_rho(self):
        return self.rho

    def run(self,init_state_tab,**kwargs):
        Starting = QuantumCircuit(
            QuantumRegister(self.ancillas,"a"),
            QuantumRegister(self.qubits,"q"),
            QuantumRegister(1,"d")
        )
        Starting.barrier()

        Starting.initialize(init_state_tab,range(self.qubits+self.ancillas+1))

        qc = Starting.compose(self.Part2)
        sc = kwargs.get("show_circ",False)
        if sc:
            fig = plt.figure(figsize = (15,15))
            ax = fig.add_subplot()
            qc.draw('mpl',ax = ax)
        simulator = AerSimulator()
        qc_comp = transpile(qc,simulator)
        res = simulator.run(qc_comp).result()
        return res.get_counts(qc_comp)
    

class op:
    up = np.array([1,0])
    down = np.array([0,1])

    @staticmethod
    def norm(state):
        return state / np.sqrt(np.conj(state) @ state)
    
    @staticmethod
    def n_fold_tp(tab):
        a = tab[0]
        for i in range(1,len(tab)):
            a = np.kron(tab[i],a)
        return a
    
    @staticmethod
    def get_initial_st(*args):
        ls = len(args[0][0])
        ans = 0
        for i in range(len(args)):
            if not (isinstance(args[i][0],str) and len(args[i][0]) == ls):
                return False
            ans += args[i][1] * op.norm(op.n_fold_tp([op.up if args[i][0][j] == "0" else op.down for j in range(len(args[i][0]))]))
        return op.norm(ans)