"""
Module: agentes/monte_carlo_agent.py
Description: Implementación de la clase base para agentes de Monte Carlo.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/24

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from agentes.tabular_agent import TabularAgent
import numpy as np
from typing import Any, Dict

class MonteCarloAgent(TabularAgent):
    """
    Clase base para agentes de Monte Carlo
    """
    
    def _init_algorithm_params(self, **kwargs):
        """
        Inicializa parámetros específicos para Monte Carlo
        
        Args:
            **kwargs: Parámetros adicionales
        """
        super()._init_algorithm_params(**kwargs)
        
        # Indica si es first-visit o every-visit Monte Carlo
        self.first_visit = kwargs.get('first_visit', True)
        
        # Para first-visit Monte Carlo
        self.visit_counts = np.zeros((self.n_states, self.n_actions))
        
        # Almacenamiento para el episodio actual
        self.episode_buffer = []
    
    def update(self, state: Any, action: int, next_state: Any, reward: float, 
               done: bool, info: Dict = None) -> None:
        """
        Guarda la transición para procesarla al final del episodio
        
        Args:
            state: Estado actual
            action: Acción tomada
            next_state: Estado siguiente
            reward: Recompensa recibida
            done: Indicador de fin de episodio
            info: Información adicional
        """
        # Guarda la transición en el buffer del episodio
        self.episode_buffer.append((state, action, reward))
        
        # Si el episodio ha terminado, procesa todo el episodio
        if done:
            self._process_episode()
    
    def _process_episode(self):
        """
        Procesa el episodio completo (a implementar en subclases)
        """
        pass
    
    def start_episode(self):
        """
        Prepara el agente para un nuevo episodio
        """
        super().start_episode()
        self.episode_buffer = []