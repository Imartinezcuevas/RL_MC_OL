"""
Module: agentes/linear_approximate_agent.py
Description: Implementación del algoritmo SARSA com aproximacion semi-gradiente.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/28

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.agent import Agent
from agentes.tile_coder import TileCoder
import numpy as np
from typing import Dict

class SARSASemiGradientAgent(Agent):
    """
    Implementación del algoritmo SARSA con aproximación semi-gradiente.
    Este es un método on-policy que aprende mientras sigue la política actual.
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para SARSA semi-gradiente
        
        Args:
            **kwargs: Parámetros adicionales
        """
        # Configuración del Tile Coding
        num_tilings = kwargs.get('num_tilings', 8)
        num_tiles = kwargs.get('num_tiles', 8)
        scale_factor = kwargs.get('scale_factor', 1.0)
        
        # Inicializar el tile coder
        self.tile_coder = TileCoder(
            self.observation_space,
            num_tilings=num_tilings,
            num_tiles=num_tiles,
            scale_factor=scale_factor
        )
        
        # Tasa de aprendizaje
        self.alpha = kwargs.get('alpha', 0.1 / num_tilings)  # Normalizado por num_tilings
        self.alpha_decay = kwargs.get('alpha_decay', 0.999)
        self.alpha_min = kwargs.get('alpha_min', 0.001)
        
        # Inicializar pesos para la aproximación lineal
        # w tendrá dimensiones [n_features, n_actions]
        self.n_actions = self.action_space.n
        self.n_features = self.tile_coder.n_features
        
        init_value = kwargs.get('init_value', 0.0)
        optimistic_init = kwargs.get('optimistic_init', False)
        
        if optimistic_init:
            self.w = np.ones((self.n_features, self.n_actions)) * init_value / num_tilings
        else:
            self.w = np.zeros((self.n_features, self.n_actions))
        
        # Para almacenar el estado y acción actual entre pasos
        self.current_state = None
        self.current_action = None
        self.current_features = None

    def get_action_values(self, state=None):
        """
        Calcula los valores Q para todas las acciones en el estado dado
        usando la aproximación lineal
        
        Args:
            state: Estado para el que calcular los valores Q
            
        Returns:
            Array con valores Q para cada acción
        """
        if state is None:
            return self.w
        
        # Codificar el estado en features
        features = self.tile_coder.encode(state)
        
        # Calcular valor Q para cada acción: Q(s,a) = sum(w_i * x_i)
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for f in features:
                q_values[a] += self.w[f, a]
        
        return q_values
    
    def get_q_value(self, features, action):
        """
        Calcula el valor Q para un conjunto de features y una acción
        
        Args:
            features: Lista de índices de features activas
            action: Acción para la que calcular el valor Q
            
        Returns:
            Valor Q
        """
        q_value = 0.0
        for f in features:
            q_value += self.w[f, action]
        return q_value
    
    def update(self, state: np.ndarray, action: int, next_state: np.ndarray, 
               reward: float, done: bool, info: Dict = None) -> None:
        """
        Actualiza los pesos usando la regla de actualización SARSA semi-gradiente
        
        Args:
            state: Estado actual
            action: Acción tomada
            next_state: Estado siguiente
            reward: Recompensa recibida
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Codificar estado actual
        features = self.tile_coder.encode(state)
        
        # Si es el primer paso del episodio, solo almacena el estado y acción
        if self.current_features is None:
            self.current_features = features
            self.current_action = action
            return
        
        # En caso contrario, actualiza los pesos con la regla semi-gradiente SARSA
        if not done:
            # Obtener siguiente acción usando la política (on-policy)
            next_action = self.get_action(next_state)
            
            # Obtener features del siguiente estado
            next_features = self.tile_coder.encode(next_state)
            
            # Calcular el valor Q' del siguiente par estado-acción
            next_q = self.get_q_value(next_features, next_action)
            
            # Calcular target: r + γQ(s',a')
            target = reward + self.gamma * next_q
        else:
            # Si es un estado terminal, el target es solo la recompensa
            target = reward
        
        # Obtener el valor Q actual
        current_q = self.get_q_value(self.current_features, self.current_action)
        
        # Calcular el error TD
        td_error = target - current_q
        
        # Actualizar los pesos: w += α * δ * ∇Q(s,a)
        # En aproximación lineal, ∇Q(s,a) = x (el vector de features)
        for f in self.current_features:
            self.w[f, self.current_action] += self.alpha * td_error
        
        # Actualizar estado y acción actual
        self.current_features = features
        self.current_action = action
    
    def start_episode(self):
        """
        Prepara el agente para un nuevo episodio
        """
        super().start_episode()
        self.current_features = None
        self.current_action = None
    
    def decay_learning_rate(self):
        """
        Reduce la tasa de aprendizaje a medida que avanza el entrenamiento
        """
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)