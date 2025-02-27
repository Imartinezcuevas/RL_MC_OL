"""
Module: agentes/approximate_agent.py
Description: Implementación de la clase base para agentes con aproximación de funciones.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/27

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.agent import Agent
import numpy as np
import gymnasium as gym

class ApproximateAgent(Agent):
    """"
    Clase base para agentes con aproximación de funciones.
    """

    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa los parámetros específicos para agentes con aproximación
        
        Args:
            **kwargs: Parámetros específicos
        """
        # Determinar si el espacio de observación es continuo o discreto
        if isinstance(self.observation_space, gym.spaces.Discrete):
            self.state_dim = self.observation_space.n
            self.is_continuous_state = False
        else:
            self.state_dim = self.observation_space.shape[0]
            self.is_continuous_state = True
            
        # Determinar si el espacio de acciones es continuo o discreto
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.n_actions = self.action_space.n
            self.is_continuous_action = False
        else:
            self.n_actions = self.action_space.shape[0]
            self.is_continuous_action = True
        
        # Tasa de aprendizaje
        self.alpha = kwargs.get('alpha', 0.01)
        self.alpha_decay = kwargs.get('alpha_decay', 0.999)
        self.alpha_min = kwargs.get('alpha_min', 0.0001)

    def decay_learning_rate(self):
        """
        Reduce la tasa de aprendizaje a medida que avanza el entrenamiento
        """
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)

    def get_features(self, state):
        """
        Extrae características del estado para la aproximación de funciones.
        Por defecto, devuelve el estado si es continuo o one-hot encoding si es discreto.
        
        Args:
            state: Estado actual
            
        Returns:
            Vector de características
        """
        if self.is_continuous_state:
            return np.array(state, dtype=np.float32)
        else:
            # One-hot encoding para estados discretos
            features = np.zeros(self.state_dim)
            features[state] = 1.0
            return features
        
    