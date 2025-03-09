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
    Agente SARSA para aprendizaje por refuerzo.
    
    Actualiza la función Q con la fórmula:
    
        Q(s, a) ← Q(s, a) + α [r + γ * Q(s', a') - Q(s, a)]
    
    donde a' se selecciona siguiendo la política epsilon-greedy.
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa los parámetros específicos para el agente SARSA.
        
        Args:
            **kwargs: Parámetros adicionales, entre ellos:
                - alpha: tasa de aprendizaje (learning rate)
        """
        # Inicializa los parámetros comunes para agentes tabulares
        super()._init_algorithm_params(**kwargs)
        # Tasa de aprendizaje
        self.alpha = kwargs.get('alpha', 0.1)
    
    def update(self, state: Any, action: int, next_state: Any, reward: float, 
               done: bool, info: Dict = None) -> None:
        """
        Actualiza la función Q en base a la transición (s, a, r, s') utilizando la fórmula de SARSA.
        
        Args:
            state: Estado actual
            action: Acción tomada en el estado actual
            next_state: Estado resultante tras la acción
            reward: Recompensa recibida
            done: Indicador de fin de episodio
            info: Información adicional del entorno (opcional)
        """
        # Si el episodio ha terminado, consideramos Q(s', a') = 0
        if done:
            td_target = reward
        else:
            # Selecciona la siguiente acción a' utilizando la política actual
            next_action = self.policy.select_action(next_state, self.Q)
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        
        # Calcula el error TD
        td_error = td_target - self.Q[state, action]
        
        # Actualiza la función Q de manera incremental
        self.Q[state, action] += self.alpha * td_error