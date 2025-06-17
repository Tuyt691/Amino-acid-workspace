#amino_acid.py contains class Amino_acid
# ---------------------------------------------
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import atom as at
import importlib
importlib.reload(at)
import math
import plotly.graph_objects as go
import numpy as np
import inspect
from sympy import *
from shared_variables import t


class Amino_acid():
    """Amino_acid class to represent an amino acid in a protein.
    Attributes:
        type (str): 3 letter code of the amino acid type (e.g., 'ALA', 'CYS').
        name (str): Name of the amino acid, representing its position in the chain.
        bb_atoms (list): List of backbone Atom objects associated with this amino acid (fixed atoms).
        sc_atoms (list): List of side-chain Atom objects associated with this amino acid.
    """
    def __init__(self, id: str, hydrogens: bool = False, structure: str = None):
        """Initializes an Amino_acid object.
            Args:
                id (str): Amino acid name (e.g., 'ALA', 'CYS').
                hydrogens: are H included in the aa
        """
        self.type = id[0:3]
        self.name = id  # Amino acid name represents its type and position in the chain, e.g. ALA1, CYS2 ...
        self.bb_atoms = self.buildBackbone()  # List of Atom objects [N, Ca, C, O, N_np1]
        self.hydrogens = hydrogens
        self.sc_atoms = self.buildSideChain()  # List of Atom objects representing side-chain atoms
        self.structure = self.structure_spe()

    @property
    def cycle_atoms(self):
        """Returns the atoms of the amino acid that are part of a cycle.
            Returns:
                list: List of Atom objects representing the atoms in the cycle.
        """
        #Complete with other cyclic amino acids
        if self.type.lower() == 'trp':
            # Tryptophan (Trp, W) has a cycle
            print('Cycle atoms:')
            for atom in [*self.sc_atoms[1:10], *self.sc_atoms[12:18]]:
                print(atom.name)
            return [*self.sc_atoms[1:10], *self.sc_atoms[12:18]]
        
        

    def __str__(self):
        """Returns a string representation of the Amino_acid object.
            Returns:
                str: String representation of the amino acid type and name.
        """
        return f"Amino_acid(type={self.type}, name={self.name}, with {len(self.bb_atoms + self.sc_atoms)} atoms.)"



    def buildBackbone(self):
        """Builds the backbone of the amino acid.
            Returns:
                list: List of Atom objects representing the backbone atoms.
        """
        # Backbone atoms are typically N, CA, C, O
        N = at.Atom(id = 'N')
        Ca = at.Atom(id = 'Ca', distance=1.45)
        C = at.Atom(id = 'C', translation= [1.45], tau=[math.radians(-70)], axis=['x'] ,distance=1.52)
        O = at.Atom(id = 'O', translation= [1.52], tau=[math.radians(-61.5)], axis=['y'] , rotable=True, distance=1.23, T_origin_nm2=C.T_origin_nm1)
        O.position  = O.position(math.radians(-45)) # For now, O is fixed
        N_np1 = at.Atom(id = 'N', translation= [1.33], tau=[math.radians(61.5)], axis=['y'] ,rotable=True, distance=1.45, T_origin_nm2=C.T_origin_nm1)
        N_np1.position  = N_np1.position(math.radians(-45)) # For now, N_np1 is fixed
        bb_atoms = [N, Ca, C, O, N_np1] # SC atoms always have the same dihedral angles configuration: phi = -57° ; psi = -47°

        return bb_atoms
    
    def buildSideChain(self):
        """Builds the side chain of the amino acid based on its type.
            
            Returns:
                list: List of Atom objects representing the side-chain atoms.
        """

        if self.type.lower() == 'gly':
            # Glycine (Gly, G) has no side chain
            return []
        
        if self.type.lower() == 'ala':
            # Alanine (Ala, A)
            Cb = at.Atom(id = 'Cb', translation= [0, 1.45], tau=[math.radians(-108), math.radians(-69)], axis=['z', 'x'] ,distance=1.52)
            return [Cb] 
        if self.type.lower() == 'trp':
            # Tryptophan (Trp, W)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            #Atomatic ring atoms angles are taken from PyMol
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(114.9-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd1 = at.Atom(id = 'Cd1', translation= [1.51], tau=[math.radians(127-180)], axis=['x'], rotable=True ,distance=1.34, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ne1 = at.Atom(id = 'Ne1', translation= [1.51, 1.34], tau=[math.radians(127-180), math.radians(69)], axis=['x', 'x'], rotable=True ,distance=1.43, 
                          T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce2 = at.Atom(id = 'Ce2', translation= [1.51, 1.34, 1.43], tau=[math.radians(127-180), math.radians(69), math.radians(76)], axis=['x', 'x', 'x'], rotable=True ,distance=1.31, 
                          T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cz2 = at.Atom(id = 'Cz2', translation= [1.51, 1.34, 1.43, 1.31], 
                          tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180)], 
                          axis=['x', 'x', 'x', 'x'], rotable=True, 
                          distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ch2 = at.Atom(id = 'Ch2', translation= [1.51, 1.34, 1.43, 1.31, 1.40],
                           tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8)],
                            axis=['x', 'x', 'x', 'x', 'x'], rotable=True, 
                            distance=1.39, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cz3 = at.Atom(id = 'Cz3', translation= [1.51, 1.34, 1.43, 1.31, 1.40, 1.39], 
                          tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(180-121.1)], 
                          axis=['x', 'x', 'x', 'x', 'x', 'x'], rotable=True ,
                          distance=1.35, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce3 = at.Atom(id = 'Ce3', translation= [1.51, 1.34, 1.43, 1.31, 1.40, 1.39, 1.35], 
                          tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(180-121.1), math.radians(180-121.2)], 
                          axis=['x', 'x', 'x', 'x', 'x', 'x', 'x'], rotable=True, 
                          distance=1.41, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cd2 = at.Atom(id = 'Cd2', translation= [1.51, 1.34, 1.43, 1.31, 1.40, 1.39, 1.35, 1.41], 
                          tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(180-121.1), math.radians(180-121.2), math.radians(180-118.3)], 
                          axis=['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'], rotable=True, 
                          distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            # Hydrogen atoms, angles taken from PyMol
            if self.hydrogens:
                Hb1 = at.Atom(id = 'Hb1', translation= [1.52, 0], tau=[math.radians(65), math.radians(109.4)], axis=['x', 'z_nm2'], rotable=True ,distance=1.1, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
                Hb2 = at.Atom(id = 'Hb2', translation= [1.52, 0], tau=[math.radians(65), math.radians(-109.4)], axis=['x', 'z_nm2'], rotable=True ,distance=1.1, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
                Hd1 = at.Atom(id = 'Hd1', translation= [1.52, 1.52], tau=[math.radians(127-180), math.radians(-56.1)], axis=['x', 'x'], rotable=True ,distance=1.1, 
                              T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                He1 = at.Atom(id = 'He1', translation= [1.52, 1.52, 1.45], tau=[math.radians(127-180), math.radians(69), math.radians(-52)], axis=['x', 'x', 'x'], rotable=True ,distance=1.1, 
                              T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                Hz2 = at.Atom(id = 'Hz2', translation= [1.52, 1.52, 1.45, 1.45, 1.52],
                               tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(120.3-180)],
                                axis=['x', 'x', 'x', 'x', 'x'], rotable=True, 
                                distance=1.1, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                Hh2 = at.Atom(id = 'Hh2', translation= [1.52, 1.52, 1.45, 1.45, 1.52, 1.52], 
                              tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(119.5-180)], 
                              axis=['x', 'x', 'x', 'x', 'x', 'x'], rotable=True ,
                              distance=1.1, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                Hz3 = at.Atom(id = 'Hz3', translation= [1.52, 1.52, 1.45, 1.45, 1.52, 1.52, 1.52], 
                              tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(180-121.1), math.radians(119.6-180)], 
                              axis=['x', 'x', 'x', 'x', 'x', 'x', 'x'], rotable=True, 
                              distance=1.1, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                He3 = at.Atom(id = 'He3', translation= [1.52, 1.52, 1.45, 1.45, 1.52, 1.52, 1.52, 1.52], 
                              tau=[math.radians(127-180), math.radians(69), math.radians(76), math.radians(129.2-180), math.radians(180-118.8), math.radians(180-121.1), math.radians(180-121.2), math.radians(120.7-180)], 
                              axis=['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'], rotable=True, 
                              distance=1.1, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
                return [Cb, Cg, Cd1, Ne1, Ce2, Cz2, Ch2, Cz3, Ce3, Cd2, Hb1, Hb2, Hd1, He1, Hz2, Hh2, Hz3, He3]


            return [Cb, Cg, Cd1, Ne1, Ce2, Cz2, Ch2, Cz3, Ce3, Cd2]
        
        if self.type.lower() == 'lys':
            # Lysine (Lys, K)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(113.4-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.52], tau=[math.radians(114.9-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce = at.Atom(id = 'Ce', translation= [1.52], tau=[math.radians(109.2-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)
            Nz = at.Atom(id = 'Nz', translation= [1.52], tau=[math.radians(116.6-180)], axis=['x'], rotable=True ,distance=1.47, T_origin_nm2=Ce.T_origin_nm1, symbolic_T_origin_nm2=Ce.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd, Ce, Nz]
        
        if self.type.lower() == 'cys':
            # Cysteine (Cys, C)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Sg = at.Atom(id = 'Sg', translation= [1.52], tau=[math.radians(115.9-180)], axis=['x'], rotable=True ,distance=1.81, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)

            return [Cb, Sg]
        
        if self.type.lower() == 'ser':
            # Serine (Ser, S)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Og = at.Atom(id = 'Og', translation= [1.52], tau=[math.radians(109.5-180)], axis=['x'], rotable=True ,distance=1.43, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)

            return [Cb, Og]
        
        if self.type.lower() == 'met':
            # Methionine (Met, M)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(109.3-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Sd = at.Atom(id = 'Sd', translation= [1.52], tau=[math.radians(110-180)], axis=['x'], rotable=True ,distance=1.81, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce = at.Atom(id = 'Ce', translation= [1.81], tau=[math.radians(100-180)], axis=['x'], rotable=True ,distance=1.78, T_origin_nm2=Sd.T_origin_nm1, symbolic_T_origin_nm2=Sd.symbolic_T_origin_nm1)

            return [Cb, Cg, Sd, Ce]
        
        if self.type.lower() == 'ile':
            # Isoleucine (Ile, I)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg1 = at.Atom(id = 'Cg1', translation= [1.52], tau=[math.radians(114.6-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cg2 = at.Atom(id = 'Cg2', translation= [1.52, 0], tau=[math.radians(114.6-180), math.radians(-120)], axis=['x', 'z_nm2'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.52], tau=[math.radians(115.9-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg1.T_origin_nm1, symbolic_T_origin_nm2=Cg1.symbolic_T_origin_nm1)

            return [Cb, Cg1, Cg2, Cd]
        
        if self.type.lower() == 'leu':
            # Leucine (Leu, L)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(118.1-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd1 = at.Atom(id = 'Cd1', translation= [1.52], tau=[math.radians(110.1-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cd2 = at.Atom(id = 'Cd2', translation= [1.52, 0], tau=[math.radians(110.1-180), math.radians(120)], axis=['x', 'z_nm2'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd1, Cd2]
        
        if self.type.lower() == 'val':
            # Valine (Val, V)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg1 = at.Atom(id = 'Cg1', translation= [1.52], tau=[math.radians(116.4-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cg2 = at.Atom(id = 'Cg2', translation= [1.52, 0], tau=[math.radians(116.4-180), math.radians(120)], axis=['x', 'z_nm2'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)

            return [Cb, Cg1, Cg2]
        
        if self.type.lower() == 'asp':
            # Aspartic acid (Asp, D)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(109.3-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Od1 = at.Atom(id = 'Od1', translation= [1.52], tau=[math.radians(125.6/2)], axis=['x'], rotable=True ,distance=1.26, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Od2 = at.Atom(id = 'Od2', translation= [1.52], tau=[math.radians(-125.6/2)], axis=['x'], rotable=True ,distance=1.26, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)

            return [Cb, Cg, Od1, Od2]
        
        if self.type.lower() == 'glu':
            # Glutamic acid (Glu, E)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(109.3-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.51], tau=[math.radians(109.5-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Oe1 = at.Atom(id = 'Oe1', translation= [1.52], tau=[math.radians(-125.6/2)], axis=['x'], rotable=True ,distance=1.26, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)
            Oe2 = at.Atom(id = 'Oe2', translation= [1.52], tau=[math.radians(125.6/2)], axis=['x'], rotable=True ,distance=1.26, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd, Oe1, Oe2]
        
        if self.type.lower() == 'arg':
            # Arginine (Arg, R)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(109.3-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.52], tau=[math.radians(109.5-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ne = at.Atom(id = 'Ne', translation= [1.52], tau=[math.radians(111-180)], axis=['x'], rotable=True ,distance=1.48, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)
            Cz = at.Atom(id = 'Cz', translation= [1.48], tau=[math.radians(123-180)], axis=['x'], rotable=True ,distance=1.33, T_origin_nm2=Ne.T_origin_nm1, symbolic_T_origin_nm2=Ne.symbolic_T_origin_nm1)
            Nh1 = at.Atom(id = 'Nh1', translation= [1.33], tau=[math.radians(-120/2)], axis=['x'], rotable=True ,distance=1.33, T_origin_nm2=Cz.T_origin_nm1, symbolic_T_origin_nm2=Cz.symbolic_T_origin_nm1)
            Nh2 = at.Atom(id = 'Nh2', translation= [1.33], tau=[math.radians(120/2)], axis=['x'], rotable=True ,distance=1.33, T_origin_nm2=Cz.T_origin_nm1, symbolic_T_origin_nm2=Cz.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd, Ne, Cz, Nh1, Nh2]
        
        if self.type.lower() == 'thr':
            # Threonine (Thr, T)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Og1 = at.Atom(id = 'Og1', translation= [1.52], tau=[math.radians(109.7-180)], axis=['x'], rotable=True ,distance=1.43, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cg2 = at.Atom(id = 'Cg2', translation= [1.52, 0], tau=[math.radians(109.7-180), math.radians(-120)], axis=['x', 'z_nm2'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)

            return [Cb, Og1, Cg2]
        
        if self.type.lower() == 'asn':
            # Asparagine (Asn, N)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(111-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Od1 = at.Atom(id = 'Od1', translation= [1.52], tau=[math.radians(-122.9/2)], axis=['x'], rotable=True ,distance=1.23, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Nd2 = at.Atom(id = 'Nd2', translation= [1.52], tau=[math.radians(122.9/2)], axis=['x'], rotable=True ,distance=1.33, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)

            return [Cb, Cg, Od1, Nd2]
        
        if self.type.lower() == 'gln':
            # Glutamine (Gln, Q)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(109.3-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.52], tau=[math.radians(111-180)], axis=['x'], rotable=True ,distance=1.52, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Oe1 = at.Atom(id = 'Oe1', translation= [1.52], tau=[math.radians(-122.9/2)], axis=['x'], rotable=True ,distance=1.23, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)
            Ne2 = at.Atom(id = 'Ne2', translation= [1.52], tau=[math.radians(122.9/2)], axis=['x'], rotable=True ,distance=1.33, T_origin_nm2=Cd.T_origin_nm1, symbolic_T_origin_nm2=Cd.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd, Oe1, Ne2]
        
        if self.type.lower() == 'phe':
            # Phenylalanine (Phe, F)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(114.9-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd1 = at.Atom(id = 'Cd1', translation= [1.51], tau=[math.radians(120-180)], axis=['x'], rotable=True ,distance=1.4, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce1 = at.Atom(id = 'Ce1', translation= [1.51, 1.40], tau=[math.radians(120-180), math.radians(180-120)], axis=['x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cz = at.Atom(id = 'Cz', translation= [1.51, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce2 = at.Atom(id = 'Ce2', translation= [1.51, 1.40, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cd2 = at.Atom(id = 'Cd2', translation= [1.51, 1.40, 1.40, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x', 'x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            return [Cb, Cg, Cd1, Ce1, Cz, Ce2, Cd2]
        
        if self.type.lower() == 'pro':
            # Proline (Pro, P)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.48)
            Cg = at.Atom(id = 'Cg', translation= [1.48], tau=[math.radians(108.4-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd = at.Atom(id = 'Cd', translation= [1.51], tau=[math.radians(106-180)], axis=['x'], rotable=True ,distance=1.50, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)

            return [Cb, Cg, Cd]
        
        if self.type.lower() == 'tyr':
            # Tyrosine (Tyr, Y)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(114.9-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Cd1 = at.Atom(id = 'Cd1', translation= [1.51], tau=[math.radians(120-180)], axis=['x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce1 = at.Atom(id = 'Ce1', translation= [1.51, 1.40], tau=[math.radians(120-180), math.radians(180-120)], axis=['x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cz = at.Atom(id = 'Cz', translation= [1.51, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce2 = at.Atom(id = 'Ce2', translation= [1.51, 1.40, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cd2 = at.Atom(id = 'Cd2', translation= [1.51, 1.40, 1.40, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120), math.radians(180-120), math.radians(180-120)], axis=['x', 'x','x', 'x', 'x'], rotable=True ,distance=1.40, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Oh = at.Atom(id = 'Oh', translation= [1.51, 1.40, 1.40, 1.40], tau=[math.radians(120-180), math.radians(180-120), math.radians(180-120), math.radians(120-180)], axis=['x', 'x','x', 'x'], rotable=True ,distance=1.36, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            return [Cb, Cg, Cd1, Ce1, Cz, Ce2, Cd2, Oh]
        
        if self.type.lower() == 'his':
            # Histidine (His, H)
            Cb = at.Atom(id = 'Cb', translation= [1.45, 0], tau=[math.radians(-69), math.radians(-120)], axis=['x', 'z_nm2'], distance=1.52)
            Cg = at.Atom(id = 'Cg', translation= [1.52], tau=[math.radians(114.9-180)], axis=['x'], rotable=True ,distance=1.51, T_origin_nm2=Cb.T_origin_nm1, symbolic_T_origin_nm2=Cb.symbolic_T_origin_nm1)
            Nd1 = at.Atom(id = 'Nd1', translation= [1.51], tau=[math.radians(180-122)], axis=['x'], rotable=True ,distance=1.39, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ce1 = at.Atom(id = 'Ce1', translation= [1.51, 1.39], tau=[math.radians(180-122), math.radians(108-180)], axis=['x', 'x'], rotable=True ,distance=1.32, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Ne2 = at.Atom(id = 'Ne2', translation= [1.51, 1.39 ,1.32], tau=[math.radians(180-122), math.radians(108-180), math.radians(109-180)], axis=['x', 'x', 'x'], rotable=True ,distance=1.31, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)
            Cd2 = at.Atom(id = 'Cd2', translation= [1.51, 1.39 ,1.32, 1.31], tau=[math.radians(180-122), math.radians(108-180), math.radians(109-180), math.radians(110-180)], axis=['x', 'x', 'x', 'x'], rotable=True ,distance=1.36, T_origin_nm2=Cg.T_origin_nm1, symbolic_T_origin_nm2=Cg.symbolic_T_origin_nm1)

            return [Cb, Cg, Nd1, Ce1, Ne2, Cd2]

        

        
    def structure_spe(self):
        if self.type.lower() in ['gly', 'ala', 'lys', 'cys', 'ser', 'met']:
            return 'linear'
        elif self.type.lower() in ['trp', 'phe', 'his', 'pro']:
            return 'aromatic'
        elif self.type.lower() in ['tyr']:
            return 'special-cyclic'
        else:
            return 'branched'
        
    def calc_fork_pos(self):
        """Calculates the position of the fork in the side chain.
        Returns:
            int: The index of the atom before the fork.
        """
        for i in range(len(self.sc_atoms) - 1):
            if self.sc_atoms[i].name[1] == self.sc_atoms[i + 1].name[1]:
                return i-1  # Return the index of the atom of the fork


    def plot_aa_static(self, dih_angles: list = None, barycentre: bool = False):
        """Plots the static representation of the amino acid.
            Args:
                For now, BB atoms are fixed
                dih_angles (list): List of dihedral angles for the side-chain atoms.
                barycentre (bool): If True, the barycentre of the amino acid is plotted.
        """
        amino_acids_static = go.Figure()
        # --- Plotting atoms ---
        # Plotting backbone atoms
        CoordinatesBB = []
        for atom in self.bb_atoms:
            atom.plot_atom(amino_acids_static)
            CoordinatesBB.append(atom.position) # List containing all bb atoms position
        # Converting the list in a np 2d float list
        CoordinatesBB = np.array(CoordinatesBB, dtype=float) # Converting into float
        CoordinatesBB = np.round(CoordinatesBB.squeeze(), 2) #Passing 2D



        # --- Plotting bonds ---
        # BB bonds
        bonds_color = 'black'
        amino_acids_static.add_trace(go.Scatter3d(
                                    x=CoordinatesBB[:-1,0], y=CoordinatesBB[:-1,1], z=CoordinatesBB[:-1,2],
                                    mode='lines',
                                    line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                    showlegend=False 
                                    ))
        
        amino_acids_static.add_trace(go.Scatter3d(
                                    x=[CoordinatesBB[2,0], CoordinatesBB[4,0]], y=[CoordinatesBB[2,1], CoordinatesBB[4,1]], z=[CoordinatesBB[2,2], CoordinatesBB[4,2]],
                                    mode='lines',
                                    line=dict(color=bonds_color, width=6),  
                                    showlegend=False  
                                    ))
        

        # --- SC Part ---
        if self.sc_atoms:
            CoordinatesSC = []
            for atom in self.sc_atoms:
                if isinstance(atom.position, np.ndarray):
                    atom.plot_atom(amino_acids_static)
                    CoordinatesSC.append(atom.position)
                else:
                    if dih_angles is None:
                        raise ValueError("Parameter 'dih_angles' must be specified to plot atoms in a specific position.")
                    params = list(inspect.signature(atom.position).parameters.keys())
                    atom.plot_atom(amino_acids_static, dih_angles[0:len(params)])
                    dih_ang = atom.send_parameters(dih_angles[0:len(params)]) #dih_ang is a dict with keys dihedral_angle1, dihedral_angle2, ... and values are angles
                    CoordinatesSC.append(atom.position(**dih_ang))
            # Converting the list in a np 2d float list
            CoordinatesSC = np.array(CoordinatesSC, dtype=float) # Converting into float
            CoordinatesSC = np.round(CoordinatesSC.squeeze(), 2) #Passing 2D
            CoordinatesSC = np.atleast_2d(CoordinatesSC)# Ensure CoordinatesSC is 2D

        #SC bonds
            #Ca - Cb
            amino_acids_static.add_trace(go.Scatter3d(
                                        x=[CoordinatesSC[0,0], CoordinatesBB[1,0]], y=[CoordinatesSC[0,1], CoordinatesBB[1,1]], z=[CoordinatesSC[0,2], CoordinatesBB[1,2]],
                                        mode='lines',
                                        line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                        showlegend=False 
                                        ))
        #Rest of linear bonds
            if self.structure_spe() in ['linear', 'aromatic']:
                if len(CoordinatesSC) > 1:
                    amino_acids_static.add_trace(go.Scatter3d(
                                    x=CoordinatesSC[:,0], y=CoordinatesSC[:,1], z=CoordinatesSC[:,2],
                                    mode='lines',
                                    line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                    showlegend=False 
                                    ))
            elif self.structure_spe() == 'branched':
                fork = self.calc_fork_pos()
                if fork >0:
                        amino_acids_static.add_trace(go.Scatter3d(
                                            x=CoordinatesSC[:fork+1,0], y=CoordinatesSC[:fork+1,1], z=CoordinatesSC[:fork+1,2],
                                            mode='lines',
                                            line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                            showlegend=False 
                                            ))
                amino_acids_static.add_trace(go.Scatter3d(
                                        x=[CoordinatesSC[fork,0], CoordinatesSC[fork+1,0]], y=[CoordinatesSC[fork,1], CoordinatesSC[fork+1,1]], z=[CoordinatesSC[fork,2], CoordinatesSC[fork+1,2]],
                                        mode='lines',
                                        line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                        showlegend=False 
                                        ))
                amino_acids_static.add_trace(go.Scatter3d(
                        x=[CoordinatesSC[fork,0], CoordinatesSC[fork+2,0]], y=[CoordinatesSC[fork,1], CoordinatesSC[fork+2,1]], z=[CoordinatesSC[fork,2], CoordinatesSC[fork+2,2]],
                        mode='lines',
                        line=dict(color=bonds_color, width=6),  # Can modify line width and color
                        showlegend=False 
                        ))
                if len(CoordinatesSC) > fork+2:
                    amino_acids_static.add_trace(go.Scatter3d(
                        x=[CoordinatesSC[-3,0], CoordinatesSC[-1,0]], y=[CoordinatesSC[-3,1], CoordinatesSC[-1,1]], z=[CoordinatesSC[-3,2], CoordinatesSC[-1,2]],
                        mode='lines',
                        line=dict(color=bonds_color, width=6),  # Can modify line width and color
                        showlegend=False 
                        ))
                
                    
            



            if self.structure_spe() == 'aromatic':
                amino_acids_static.add_trace(go.Scatter3d(
                                        x=[CoordinatesSC[1,0], CoordinatesSC[-1,0]], y=[CoordinatesSC[1,1], CoordinatesSC[-1,1]], z=[CoordinatesSC[1,2], CoordinatesSC[-1,2]],
                                        mode='lines',
                                        line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                        showlegend=False 
                                        ))
                if self.type.lower() == 'trp':
                    amino_acids_static.add_trace(go.Scatter3d(
                                                    x=[CoordinatesSC[4,0], CoordinatesSC[9,0]], y=[CoordinatesSC[4,1], CoordinatesSC[9,1]], z=[CoordinatesSC[4,2], CoordinatesSC[9,2]],
                                                    mode='lines',
                                                    line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                                    showlegend=False 
                                                    ))

                # if self.hydrogens:
                #     # amino_acids_static.add_trace(go.Scatter3d(
                #     #                             x=CoordinatesSC[:-8,0], y=CoordinatesSC[:-8,1], z=CoordinatesSC[:-8,2],
                #     #                             mode='lines',
                #     #                             line=dict(color=bonds_color, width=6),  # Can modify line width and color
                #     #                             showlegend=False 
                #     #                             ))
                #     amino_acids_static.add_trace(go.Scatter3d(
                #                                 x=[CoordinatesSC[0,0], CoordinatesSC[10,0]], y=[CoordinatesSC[0,1], CoordinatesSC[10,1]], z=[CoordinatesSC[0,2], CoordinatesSC[10,2]],
                #                                 mode='lines',
                #                                 line=dict(color=bonds_color, width=6),  # Can modify line width and color
                #                                 showlegend=False 
                #                                 ))
                #     amino_acids_static.add_trace(go.Scatter3d(
                #                                 x=[CoordinatesSC[0,0], CoordinatesSC[11,0]], y=[CoordinatesSC[0,1], CoordinatesSC[11,1]], z=[CoordinatesSC[0,2], CoordinatesSC[11,2]],
                #                                 mode='lines',
                #                                 line=dict(color=bonds_color, width=6),  # Can modify line width and color
                #                                 showlegend=False 
                #                                 ))
                    
            if self.type.lower() == 'pro':
                amino_acids_static.add_trace(go.Scatter3d(
                                            x=[CoordinatesBB[0,0], CoordinatesSC[-1,0]], y=[CoordinatesBB[0,1], CoordinatesSC[-1,1]], z=[CoordinatesBB[0,2], CoordinatesSC[-1,2]],
                                            mode='lines',
                                            line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                            showlegend=False 
                                            ))
            
            if self.type.lower() == 'tyr':
                amino_acids_static.add_trace(go.Scatter3d(
                                    x=CoordinatesSC[:-1,0], y=CoordinatesSC[:-1,1], z=CoordinatesSC[:-1,2],
                                    mode='lines',
                                    line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                    showlegend=False 
                                    ))
                amino_acids_static.add_trace(go.Scatter3d(
                                            x=[CoordinatesSC[1,0], CoordinatesSC[6,0]], y=[CoordinatesSC[1,1], CoordinatesSC[6,1]], z=[CoordinatesSC[1,2], CoordinatesSC[6,2]],
                                            mode='lines',
                                            line=dict(color=bonds_color, width=6),  # Can modify line width and color
                                            showlegend=False 
                                            ))
                amino_acids_static.add_trace(go.Scatter3d(
                            x=[CoordinatesSC[4,0], CoordinatesSC[-1,0]], y=[CoordinatesSC[4,1], CoordinatesSC[-1,1]], z=[CoordinatesSC[4,2], CoordinatesSC[-1,2]],
                            mode='lines',
                            line=dict(color=bonds_color, width=6),  # Can modify line width and color
                            showlegend=False 
                            ))
                
        amino_acids_static.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',  # Preserves the true aspect ratio of your data
                ),
                width=1000,  # Increased width (pixels)
                height=800,  # Increased height (pixels)
                margin=dict(l=0, r=0, b=0, t=50),  # Reduced margins to maximize plot area
                template="plotly_white"  # Clean white background template
                )

        return amino_acids_static
    
    def from_sympy_to_lambda(self, func):
        """Converts a sympy function to a lambda function.
            Args:
                func (sympy function): The sympy function to convert.
            Returns:
                function: The converted lambda function.
        """
        # Convert the sympy expression to a lambda function
        parameters = func.free_symbols
        # print(parameters)
        func_lambda = lambdify(parameters, func, 'numpy')
        return func_lambda
    
    def plot_workspace(self, static_dih_ang: list = None, len_dih_angles: list = None):
        """Plots the workspace of the amino acid.
            Args:
                len_dih_angles (int): Length of the dihedral angles vectors to plot.
                static_dih_ang (list): List of dihedral angles to plot the static representation.
        """
        if len_dih_angles is None:
            if self.type.lower() == 'gly':
                print("Glycine has no side chain, so the workspace is not defined. Returning the static plot instead.")
                return self.plot_aa_static()
            elif self.type.lower() == 'ala':
                print("Alanine has a side chain, but no dihedral angles to plot. Returning the static plot instead.")
                return self.plot_aa_static()
            elif self.type.lower() == 'pro':
                print("Proline has a unique structure with no dihedral angles to plot.")
                return None
            else:
                raise ValueError("Parameter 'len_dih_angles' must be specified to plot the workspace. It is the number of points in each dihedral angle of the aa")
        # Create a figure
        Workspace = self.plot_aa_static(dih_angles=static_dih_ang)

        # Generate a grid of dihedral angles
        dih_ang = {f"dihedral_angle{i+1}": np.linspace(-math.pi, math.pi, val) for i, val in enumerate(len_dih_angles)}
        #dih_ang is a dict with keys dihedral_angle1, dihedral_angle2, ... and values are lists of angles
        ws_positions = []
        for angle1 in dih_ang['dihedral_angle1']:
            for atom in self.sc_atoms:
                if callable(atom.position):
                        params = list(inspect.signature(atom.position).parameters.keys())
                        if len(params) == 1:
                            ws_positions.append(atom.position(angle1))
                        elif len(params) == 2:
                            for angle2 in dih_ang['dihedral_angle2']:
                                ws_positions.append(atom.position(angle1, angle2))
                        elif len(params) == 3:
                            for angle2 in dih_ang['dihedral_angle2']:
                                for angle3 in dih_ang['dihedral_angle3']:
                                    ws_positions.append(atom.position(angle1, angle2, angle3))
                        elif len(params) == 4:
                            for angle2 in dih_ang['dihedral_angle2']:
                                for angle3 in dih_ang['dihedral_angle3']:
                                    for angle4 in dih_ang['dihedral_angle4']:
                                        ws_positions.append(atom.position(angle1, angle2, angle3, angle4))
                        elif len(params) == 5:
                            for angle2 in dih_ang['dihedral_angle2']:
                                for angle3 in dih_ang['dihedral_angle3']:
                                    for angle4 in dih_ang['dihedral_angle4']:
                                        for angle5 in dih_ang['dihedral_angle5']:
                                            ws_positions.append(atom.position(angle1, angle2, angle3, angle4, angle5))
                        
        #Converting the lit into a (n,3) nparray
        ws_positions = np.array(ws_positions)
        # Reshaping the data
        ws_positions = np.squeeze(ws_positions, axis=2)
        # Create a color scale based on the position in the array
        colors = np.linspace(0, 1, len(ws_positions))
        
        Workspace.add_trace(go.Scatter3d(
                x=ws_positions[:,0], y=ws_positions[:,1], z=ws_positions[:,2],
                mode='markers',
                textposition="top center",
                marker=dict(size=2, color=colors, colorscale='Greens'),
                showlegend=False  # Deactivate legend
            ))
        
        Workspace.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data',  # Preserves the true aspect ratio of your data
                ),
                width=1000,  # Increased width (pixels)
                height=800,  # Increased height (pixels)
                margin=dict(l=0, r=0, b=0, t=50),  # Reduced margins to maximize plot area
                template="plotly_white"  # Clean white background template
                )
        
        return Workspace
    
    def symb_cycle_barycentre(self, *atoms):
        """
        Calculate the barycentre of a molecule composed of a variable number of atoms.

        Parameters:
        *atoms: Variable length argument list of atoms. Each atom is a tuple (mass, (x, y, z)).

        Returns:
        tuple: The coordinates of the barycentre (x, y, z), x, y and z are sympy expressions.
        """
        total_mass = 0
        total_symb_mass = 0
        weighted_sum_x = 0
        weighted_sum_y = 0
        weighted_sum_z = 0

        

        for atom in atoms:
            mass = atom.mass
            symb_mass = atom.symbolic_mass
            position = atom.symbolic_position
            total_mass += mass
            total_symb_mass += symb_mass
            weighted_sum_x += mass * position[0]
            weighted_sum_y += mass * position[1]
            weighted_sum_z += mass * position[2]

        if total_mass == 0:
            raise ValueError("Total mass of the molecule cannot be zero.")

        barycentre_x = weighted_sum_x / total_mass
        barycentre_y = weighted_sum_y / total_mass
        barycentre_z = weighted_sum_z / total_mass

        # Simplify the barycentre coordinates
        barycentre_x = barycentre_x.n(3)
        barycentre_y = barycentre_y.n(3)
        barycentre_z = barycentre_z.n(3)

        self.cycle_mass = total_symb_mass

        return Matrix([barycentre_x, barycentre_y, barycentre_z]) # Sympy expressions !
    
    def velocity(self, position):
        """Calculates the velocity of the amino acid based on its position.
            Args:
                position (list): Position of the instance.
            Returns:
                list: Velocity of the instance.
        """
        velocity = diff(position, t)
        dtdχ_vars = [diff(sym, t) for sym in velocity.atoms(Function) if str(sym).startswith('χ')]
        factored_velocity = factor(velocity, *dtdχ_vars) # doesn't works, can improve this
        return factored_velocity