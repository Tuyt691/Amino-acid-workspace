# atome.py contains class Atom
# ---------------------------------------------
import numpy as np
from sympy import *
import inspect
import math
import plotly.graph_objects as go
from shared_variables import t

class Atom():
    """Atom class to represent a chemical atom in a protein.
    Attributes:
        type (str): Type of the atom (e.g., 'C', 'H').
        name (str): Name of the atom, representing chemical type and position in the chain.
        position (nparray of lambda functions): Position of the atom in 3D space.
        T_origin_nm1 (nparray of lambda functions): Transformation matrix from the origin to n-1.
        symbolic_position (sympy.Matrix): Symbolic position of the atom.
        symbolic_T_origin_nm1 (sympy.Matrix): Symbolic transformation matrix from the origin to n-1.
    """
    # # Class attribute
    # t = symbols('t') # t variable for the symbolic position of the atom

    def __init__(self, id: str,
                 translation: list = [0], tau: list = [0], axis: list = [None], rotable: bool = False, distance: int = 0, T_origin_nm2= np.identity(4),
                 symbolic_T_origin_nm2: Matrix = np.identity(4)):
        """Initializes an Atom object.
            Args:
                id (str): Atom name (e.g., 'C1', 'H2').
                translation (int list): Distance from atom n-2 to atom n-1.
                tau (list of int): Oriented angle between atoms n-2, n-1 and n (in radians).
                axis (str list): Rotation axis for tau angle ('x', 'y', 'z').
                rotable (bool): Is there a dihedral angle influencing this atom's position.
                distance (int): Distance from Atom to n-1.
                T_origin_nm2: Transformation matrix from atom n-2 to n-1.
                symbolic_T_origin_nm1: Symbolic transformation matrix from origin to atom n-2.
        """
        self.type = id[0]
        self.name = id  # Atom name tepresent its position in the chain, e.g. C1, C2 ...
        info_pos = self.PositionCalculation(translation, tau, axis, rotable, distance, T_origin_nm2)
        self.position = info_pos[0]  # Position of the atom in 3D space, /!\ lambda function
        self.T_origin_nm1 = info_pos[1]  # Transformation matrix from the origin to n-1
        self.mass = self.set_mass(self.type)  # Mass of the atom in kg
        # Some private attribute to be accessed by optional calculations
        self._translation = translation
        self._tau = tau
        self._axis = axis
        self._rotable = rotable
        self._T_origin_nm2 = T_origin_nm2
        self._distance = distance
        self._symbolic_T_origin_nm2 = symbolic_T_origin_nm2
        self._sym_attributes = None  # Placeholder for symbolic position, if needed

    
    @property
    def symbolic_position(self):
        return self._get_sym_calculation_result()[0]

    @property
    def symbolic_T_origin_nm1(self):
        return self._get_sym_calculation_result()[1]
    
    @property
    def symbolic_mass(self):
        return self.set_sym_mass(self.type)
    
    @property
    def symbolic_velocity(self):
        """Returns the symbolic velocity of the atom."""
        return self.symb_vel_calc()
    
    def _get_sym_calculation_result(self):
        # Returns the symbolic position of the atom
        if self._sym_attributes is None:
            self._sym_attributes = self._sym_calculation()
        return self._sym_attributes

    # ------------------------------------------------------------------------------------------------------------
    def _sym_calculation(self):
        #Calculates the symbolic position of the atom

        def get_existing_vars(sym, var_name: str):
            """Get existing variable names from a symbolic expression."""
            if var_name is None:
                raise ValueError("You must specify a variable to search for.")
            functions = set()
            for element in sym:
                functions.update(element.atoms(Function)) # atoms is a sympy function
            func_of_interest = [f"{func}" for func in functions if func.func.__name__.startswith(var_name)]
            return func_of_interest
        
        def get_max_index(var_list: list):
            """Get the maximum index from a list of variable names."""
            if var_list:
                indices = [int(var[2]) for var in var_list] # Extract the number of the variable
                return max(indices)
            return 0
        
        # Handle the case where Sym_T_origin_nm2 is None or a numpy array
        if np.array_equal(self._symbolic_T_origin_nm2, np.identity(4)):
            if not np.array_equal(self._T_origin_nm2, np.eye(4)):
                # If _T_origin_nm2 is a function, then a _symbolic_T_origin_nm2 must be specified
                raise ValueError("You must specify a symbolic transformation matrix that goes from the origin to atom n-2.")
            existing_chi_vars  = []
        else:
            # Extract existing tau variables from self._T_origin_nm2
            existing_chi_vars = get_existing_vars(self._symbolic_T_origin_nm2, 'χ_')
        # Determine the next available index for new chi variables
        max_index_chi = get_max_index(existing_chi_vars)

        # Create new tau variables with the correct names
        new_chi_vars = Function(f'χ_{max_index_chi+1}')(t)  # Create a new symbolic variable for the dihedral angle, function of t
        chi_symbols = existing_chi_vars + [new_chi_vars]
        # Create the symbolic transformation matrix
        # ------------------------------
        # Ensures rotations around bond angles and translations
        T_nmp_nm1 = np.eye(4)

        for i in range(len(self._axis)):

            if self._axis[i] == 'x':
                T_nm2_nm1 = Matrix([[1, 0, 0, 0],
                                      [0, cos(self._tau[i]), -sin(self._tau[i]), 0],
                                      [0, sin(self._tau[i]), cos(self._tau[i]), self._translation[i]],
                                      [0, 0, 0, 1]])
            elif self._axis[i] == 'y':
                T_nm2_nm1 = Matrix([[cos(self._tau[i]), 0, sin(self._tau[i]), 0],
                                      [0, 1, 0, 0],
                                      [-sin(self._tau[i]), 0, cos(self._tau[i]), self._translation[i]],
                                      [0, 0, 0, 1]])
            elif self._axis[i] =='z':
                T_nm2_nm1 = Matrix([[cos(self._tau[i]), -sin(self._tau[i]), 0, 0],
                                      [sin(self._tau[i]), cos(self._tau[i]), 0, 0],
                                      [0, 0, 1, self._translation[i]],
                                      [0, 0, 0, 1]])
            elif self._axis[i] == 'z_nm2': # Assuming this is the last rotation
                T_nm2_nm1 = Matrix([[cos(self._tau[i]), -sin(self._tau[i]), 0, 0],
                                      [sin(self._tau[i]), cos(self._tau[i]), 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
                
                T_nmp_nm1 = T_nm2_nm1 @ T_nmp_nm1 # We want to do a non parametrized rotation along self.axis z_nm2
                break # We assume this is the last rotation
            
            elif self._axis[i] == None: # for the N and Ca case
                T_nm2_nm1 = Matrix([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
                
            T_nmp_nm1 =T_nmp_nm1 @ T_nm2_nm1


        # ------------------------------
        # Parametrized chi rotations
        if self._rotable:
            # Pi/2 deleted
            # Create the transformation matrix with the dynamic parameter name
            Transfo_mat_chi = Matrix([[cos(+chi_symbols[-1]), -sin(+chi_symbols[-1]), 0, 0],
                                        [sin(+chi_symbols[-1]), cos(+chi_symbols[-1]), 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]
                                        ])
    
            # Multiply matrices
            T_nmp_nm1 = Transfo_mat_chi @ T_nmp_nm1
        # ------------------------------

        T_origin_nm1 = self._symbolic_T_origin_nm2 @ T_nmp_nm1

        # Create the symbolic coordinates of the atom
        coord_atom = Matrix([[0],
                             [0],
                             [self._distance],
                             [1]])
        coord_atom_sym = T_origin_nm1 @ coord_atom
        coord_atom_sym = coord_atom_sym.n(3)
        coord_atom_sym.row_del(3)

        return [coord_atom_sym,
                T_origin_nm1.n(3)] # Type = sympy variable
    
    def symb_vel_calc(self):
        """Calculates the symbolic velocity of the atom."""
        # Get the symbolic position of the atom
        sym_position = self.symbolic_position

        # Differentiate the position function with respect to t
        velocity = diff(sym_position, t)

        return velocity
    
    
    # ------------------------------------------------------------------------------
    
    # -------- This part is dedicated to the numerical position of the atom --------


    def __str__(self):
        """String representation of the Atom object."""
        return f"Atom(type={self.type}, name={self.name}, position={self.position})"
    

    def get_required_params(self, matrix_func):
        """Get the required parameters for a matrix function.
            Args:
                matrix_func: The matrix function to inspect.
            Returns:
                list: List of required parameter names.
        """
        return list(inspect.signature(matrix_func).parameters.keys())

    def multiply_matrix_lambdas(self, f1, f2):
        """Multiply two matrices while preserving original parameter names.
        Args:
            f1: First matrix (function or numpy array).
            f2: Second matrix (function or numpy array).
        Returns:
            function: A function representing the product of the two matrix functions
            with preserved parameter names.
        """

        #Case 1: both imputs are numpy arrays
        if isinstance(f1, np.ndarray) and isinstance(f2, np.ndarray):
            return f1 @ f2 
        
        #Case 2: first input is numpy array, second is a function
        elif isinstance(f1, np.ndarray) and callable(f2):

            # Create a function with the same parameter names as f2
            def result_func(*args, **kwargs):
                return f1 @ f2(*args, **kwargs)

            # Set the signature of the new function to match f2
            result_func.__signature__ = inspect.signature(f2)
            return result_func
        
        # Case 3: First input is a function, second is numpy array
        elif callable(f1) and isinstance(f2, np.ndarray):

            # Create a function with the same parameter names as f1
            def result_func(*args, **kwargs):
                return f1(*args, **kwargs) @ f2

            # Set the signature of the new function to match f1
            result_func.__signature__ = inspect.signature(f1)
            return result_func

        # Case 4: Both inputs are functions
        else:
            f1_sig = inspect.signature(f1)
            f2_sig = inspect.signature(f2)

            # Merge the parameter lists from both functions
            all_params = list(f1_sig.parameters.values()) + [
                param for name, param in f2_sig.parameters.items()
                if name not in f1_sig.parameters
            ]

            # Create a combined signature
            combined_sig = inspect.Signature(parameters=all_params)

            # Create a function that accepts all parameters from both f1 and f2
            def result_func(*args, **kwargs):
                # Extract parameters for f1
                f1_kwargs = {k: v for k, v in kwargs.items() if k in f1_sig.parameters}
                # Extract parameters for f2
                f2_kwargs = {k: v for k, v in kwargs.items() if k in f2_sig.parameters}

                # Handle positional arguments if any
                if args:
                    # Assign positional args to the appropriate functions based on their signatures
                    f1_arg_count = len(f1_sig.parameters)
                    f1_args = args[:f1_arg_count]
                    f2_args = args[f1_arg_count:]

                    return f1(*f1_args, **f1_kwargs) @ f2(*f2_args, **f2_kwargs)
                else:
                    return f1(**f1_kwargs) @ f2(**f2_kwargs)

            # Assign the combined signature to the new function
            result_func.__signature__ = combined_sig
            return result_func
        
    # Function to get the next available dihedral angle parameter name
    def get_next_dihedral_param_name(self, existing_matrix):
        """
        Get the next available dihedral_angle{n} parameter name based on an existing matrix.

        Args:
            existing_matrix: The existing matrix function to inspect for parameter names.

        Returns:
            str: The next available dihedral angle parameter name (e.g., 'dihedral_angle3')
        """
        if existing_matrix is None:
            return 'dihedral_angle1'

        # Get parameters from existing matrix if it's a function
        if callable(existing_matrix):
            params = self.get_required_params(existing_matrix)
        else:
            return 'dihedral_angle1'  # If not callable, start with 1

        # Find parameters that match the pattern 'dihedral_angle{n}'
        dihedral_params = [p for p in params if p.startswith('dihedral_angle')]

        # Extract the numbers from the parameter names
        used_numbers = []
        for param in dihedral_params:
            try:
                num = int(param.replace('dihedral_angle', ''))
                used_numbers.append(num)
            except ValueError:
                continue
            
        # Find the next available number
        if not used_numbers:
            next_num = 1
        else:
            next_num = max(used_numbers) + 1
        return f'dihedral_angle{next_num}'
    
    # Create a lambda function dynamically using the parameter name
    def create_transfo_mat_with_param(self, param_name):
        """Create a transformation matrix with the given parameter name."""
        # Create the function signature dynamically
        exec_globals = {
            'np': np,
            'math': math,
        }
        # /!\ I have a big doubt on this pi/2 offset !!!
        exec(f"def dynamic_transfo_mat({param_name}):\n"
                f"    return np.array([\n"
                f"        [np.cos({param_name}), -np.sin({param_name}), 0, 0],\n"
                f"        [np.sin({param_name}), np.cos({param_name}), 0, 0],\n"
                f"        [0, 0, 1, 0],\n"
                f"        [0, 0, 0, 1]\n"
                f"    ])", exec_globals)

        return exec_globals['dynamic_transfo_mat']


    # Calculates atoms position
    def PositionCalculation(self, translation: list, tau: list, axis: list, rotable: bool, distance: float, T_origin_nm2): #return also change of basis matrix
        """Calculates the position of the atom based on the given parameters.
             Args:
                translation (float list): Distance from atom n-2 to atom n-1.
                tau (in list): Oriented angle between atoms n-2, n-1 and n (in radians).
                axis (str list): Rotation axis for tau angle ('x', 'y', 'z').
                rotable (bool): Is there a dihedral angle influencing this atom's position.
                distance (float): Distance from Atom to n-1.
                T_origin_nm2: Transformation matrix from atom n-2 to n-1.
                """
        # ------------------------------
        # This first part ensures basic nonparametrized rotations:
        #With successive multiplications, the rotation number is defined by the length of the axis] vector
        #if axis = ['x', 'z_nm2'], we want to do a non parametrized rotation along axis z_nm2

        T_nmp_nm1 = np.eye(4)

        for i in range(len(axis)):
            if axis[i] == 'x':
                T_nm2_nm1 = np.array([[1, 0, 0, 0],
                                      [0, np.cos(tau[i]), -np.sin(tau[i]), 0],
                                      [0, np.sin(tau[i]), np.cos(tau[i]), translation[i]],
                                      [0, 0, 0, 1]])
            elif axis[i] == 'y':
                T_nm2_nm1 = np.array([[np.cos(tau[i]), 0, np.sin(tau[i]), 0],
                                      [0, 1, 0, 0],
                                      [-np.sin(tau[i]), 0, np.cos(tau[i]), translation[i]],
                                      [0, 0, 0, 1]])
            elif axis[i] =='z':
                T_nm2_nm1 = np.array([[np.cos(tau[i]), -np.sin(tau[i]), 0, 0],
                                      [np.sin(tau[i]), np.cos(tau[i]), 0, 0],
                                      [0, 0, 1, translation[i]],
                                      [0, 0, 0, 1]])
            else: # so if axis[i] == 'z_nm2'
                T_nm2_nm1 = np.array([[np.cos(tau[i]), -np.sin(tau[i]), 0, 0],
                                      [np.sin(tau[i]), np.cos(tau[i]), 0, 0],
                                      [0, 0, 1, translation[i]],
                                      [0, 0, 0, 1]])
                
                T_nmp_nm1 = T_nm2_nm1 @ T_nmp_nm1 # We want to do a non parametrized rotation along axis z_nm2
                break # We assume this is the last rotation
                
            T_nmp_nm1 =T_nmp_nm1 @ T_nm2_nm1
        # ------------------------------

        # ------------------------------
        #This second part ensure parametrized rotations
        if rotable:
            # Get the next available dihedral angle parameter name if first rotation, get 'dihedral_angle1
            param_name = self.get_next_dihedral_param_name(T_origin_nm2)
    
            # Create the transformation matrix with the dynamic parameter name
            Transfo_mat_chi = self.create_transfo_mat_with_param(param_name)
    
            # Multiply matrices
            T_nmp_nm1 = self.multiply_matrix_lambdas(Transfo_mat_chi, T_nmp_nm1)
        # ------------------------------

        T_origin_nm1 = self.multiply_matrix_lambdas(T_origin_nm2, T_nmp_nm1)

        coord_atom = np.array([[0],
                               [0],
                               [distance],
                               [1]])
        
        coord_atom_ref = self.multiply_matrix_lambdas(T_origin_nm1,coord_atom)

        if rotable:
            return [self.multiply_matrix_lambdas(np.eye(4)[:3], coord_atom_ref),
                    T_origin_nm1] # Type = lambda function
        
        else:
            return [np.round(np.array(coord_atom_ref[0:3], dtype=float), 2), 
                    np.array(T_origin_nm1)]
        
    def send_parameters(self, dih_angles: list):
        """Return the dihedral angles of the atom in adictionnary.
            Returns:
                dict: Dictionary of dihedral angles.
        """
        return {f"dihedral_angle{i+1}": val for i, val in enumerate(dih_angles)}
        
    def plot_atom(self, figure, angles: list = None):
        """Plots the atom in 3D space using Plotly.
            Args:
                figure: The Plotly figure to add the atom to.
        """
        def sel_color(self):
            """Returns the color of the atom based on its type."""
            if self.type == 'C':
                return 'black'
            elif self.type == 'N':
                return 'blue'
            elif self.type == 'O':
                return 'red'
            elif self.type == 'H':
                return 'white'
            elif self.type == 'S':
                return 'yellow'
            else:
                return 'grey'

        def plot(x, y, z):
            figure.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                textposition="top center",
                marker=dict(size=7, color=sel_color(self)),
                showlegend=False  # Deactivate legend
            ))

        if angles == None: 
            x = float(self.position[0])
            y = float(self.position[1])
            z = float(self.position[2])
            plot(x, y, z)
        else:
            dih_ang = self.send_parameters(angles)
            x = float(self.position(**dih_ang)[0])
            y = float(self.position(**dih_ang)[1])
            z = float(self.position(**dih_ang)[2])
            plot(x, y, z)

    def set_mass(self, type: str):
        """Set the mass of the atom based on its type.
            The mass is given in kg.
            Args:
                type (str): Type of the atom (e.g., 'C', 'H').
            Returns:
                mass in kg. (m(kg) = 1.66053906660e-27 * mass(amu))"""
        if type == 'C':
            return 19.9260e-27
        elif type == 'N':
            return 23.2519e-27
        elif type == 'O':
            return 26.5595e-27
        elif type == 'H':
            return 1.6734e-27
        
    def set_sym_mass(self, type: str):
        if type == 'C':
            return symbols('m_C')
        elif type == 'N':
            return symbols('m_N')
        elif type == 'O':
            return symbols('m_O')
        elif type == 'H':
            return symbols('m_H')

