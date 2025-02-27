"""
Module: agentes/linear_approximate_agent.py
Description: Implementación del algoritmo SARSA com aproximacion semi-gradiente.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/27

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.linear_approximate_agent import LinearApproximateAgent
from typing import Any, Dict

class SARSASemiGradientAgent(LinearApproximateAgent):
    """
    Implementación del algoritmo SARSA con aproximación semi-gradiente.
    Este es un método on-policy que aprende mientras sigue la política actual.
    """
    
    def update(self, state: Any, action: int, next_state: Any, reward: float, 
               done: bool, info: Dict = None) -> None:
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
        # Obtiene el valor Q actual
        current_q = self.get_q_value(state, action)
        
        # Selecciona la siguiente acción según la política actual (SARSA es on-policy)
        next_action = self.get_action(next_state) if not done else None
        
        # Calcula el valor objetivo para la actualización
        if done:
            target = reward
        else:
            next_q = self.get_q_value(next_state, next_action)
            target = reward + self.gamma * next_q
        
        # Calcula el error delta (diferencia temporal)
        delta = target - current_q
        
        # Obtiene el vector de características del estado
        features = self.get_features(state)
        
        # Actualiza los pesos en la dirección del gradiente
        # La derivada de Q(s,a) = w • x con respecto a w es simplemente x
        self.weights[action] += self.alpha * delta * features