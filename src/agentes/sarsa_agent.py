"""
Module: agentes/sarsa_agent.py
Description: Implementación de la clase base para agentes SARSA.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/26

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.tabular_agent import TabularAgent
import numpy as np
from typing import Any, Dict

class SARSAAgent(TabularAgent):
    """
    Implementación del algoritmo SARSA (State-Action-Reward-State-Action)
    on-policy.
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para SARSA
        
        Args:
            **kwargs: Parámetros adicionales
        """
        super()._init_algorithm_params(**kwargs)
        
        # Tasa de aprendizaje
        self.alpha = kwargs.get('alpha', 0.1)
        
        # Para exploración decreciente
        self.min_alpha = kwargs.get('min_alpha', 0.01)
        
        # Para almacenar el estado y acción actual entre pasos
        self.current_state = None
        self.current_action = None
    
    def update(self, state: Any, action: int, next_state: Any, reward: float, 
               done: bool, info: Dict = None) -> None:
        """
        Actualiza los valores Q usando la regla de actualización SARSA
        
        Args:
            state: Estado actual
            action: Acción tomada
            next_state: Estado siguiente
            reward: Recompensa recibida
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Si es el primer paso del episodio, solo almacena el estado y acción
        if self.current_state is None:
            self.current_state = state
            self.current_action = action
            return
        
        # En caso contrario, actuliza la funcion Q con la regla SARSA
        next_action = self.get_action(next_state) if not done else 0

        # Obtiene los valores actuales
        current_q = self.Q[self.current_state, self.current_action]

        # Calculamos la recompensa para la actualizacion
        if done:
            # Si el episodio ha terminado, no hay estado siguiente
            # Si el estado es terminal, Q(St+1,At+1) se define como cero
            # La actualización se convierte en R + γ*0 = R
            target = reward
        else:
            # Si no ha terminado, usamos el valor Q del siguiente par estado accion
            target = reward + self.gamma * self.Q[next_state, next_action]

        # Actualizamos Q
        self.Q[self.current_state, self.current_action] += self.alpha * (target - current_q)

        # Actualizamos el estado y accion actual para el siguiente paso
        self.current_action = next_action
        self.current_state = next_state
    
    def start_episode(self):
        """
        Prepara el agente para un nuevo episodio
        """
        super().start_episode()
        self.current_state = None
        self.current_action = None

    def decay_learning_rate(self, episode, total_episodes):
        """
        Reduce la tasa de aprendizaque a medida que avanza el entrenamiento

        Args:
            episode: Episodio actual
            total_episodes: Número total de episodios
        """

        self.alpha = max(self.min_alpha, self.alpha * (1 - episode / total_episodes))