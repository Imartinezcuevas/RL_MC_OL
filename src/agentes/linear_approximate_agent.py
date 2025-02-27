"""
Module: agentes/linear_approximate_agent.py
Description: Implementación de la clase base para agentes con aproximación lineal de la funcion Q.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/27

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.approximate_agent import ApproximateAgent
import numpy as np

class LinearApproximateAgent(ApproximateAgent):
    """Clase base para agentes con aproximación lineal de la función Q"""
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa los parámetros específicos para agentes con aproximación lineal
        
        Args:
            **kwargs: Parámetros específicos
        """
        super()._init_algorithm_params(**kwargs)
        
        # Inicializa los pesos para cada acción
        self.weights = np.zeros((self.n_actions, self.state_dim))
        
    def get_q_value(self, state, action):
        """
        Calcula el valor Q para un estado y acción específicos usando aproximación lineal
        
        Args:
            state: Estado actual
            action: Acción a evaluar
            
        Returns:
            Valor Q estimado
        """
        features = self.get_features(state)
        return np.dot(self.weights[action], features)
    
    def get_action_values(self):
        """
        Devuelve la función aproximadora de Q (en este caso, los pesos)
        
        Returns:
            Pesos de la aproximación
        """
        return self.weights
    
    def get_all_q_values(self, state):
        """
        Calcula los valores Q para todas las acciones en un estado dado
        
        Args:
            state: Estado actual
            
        Returns:
            Valores Q para todas las acciones
        """
        features = self.get_features(state)
        return np.dot(self.weights, features)