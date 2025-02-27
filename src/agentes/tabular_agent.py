"""
Module: agentes/tabular_agent.py
Description: Implementación de la clase base para agentes con representación tabular.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.agent import Agent
import numpy as np
import gymnasium as gym

class TabularAgent(Agent):
    """Clase base para agentes con representación tabular de la función Q"""
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa los parámetros específicos para agentes tabulares
        
        Args:
            **kwargs: Parámetros específicos
        """  
        self.n_states = self.observation_space.n
        self.n_actions = self.action_space.n
        
        # Inicializa la tabla Q
        # Inicializa la tabla Q
        init_value = kwargs.get('init_value', 0.0)
        optimistic_init = kwargs.get('optimistic_init', False)

        if optimistic_init:
            # Inicialización optimista para fomentar la exploración
            self.Q = np.ones((self.n_states, self.n_actions)) * init_value
        else:
            # Inicialización a cero o valor específico
            self.Q = np.zeros((self.n_states, self.n_actions))
        
    
    def get_action_values(self):
        """
        Devuelve la tabla Q
        
        Returns:
            Tabla Q
        """
        return self.Q