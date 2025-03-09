"""
Module: agentes/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete agentes.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# Importación de módulos o clases
from .agent import Agent
from .tabular_agent import TabularAgent
from .monte_carlo_agent import MonteCarloAgent
from .monte_carlo_off_policy_agent import MonteCarloOffPolicyAgent
from .monte_carlo_on_policy_agent import MonteCarloOnPolicyAgent
from .sarsa_agent import SARSAAgent
from .qlearning_agent import QLearningAgent
from .sarsa_semigradient_agent import SARSASemiGradientAgent
from .dqlearning_agent import DeepQAgent

# Lista de módulos o clases públicas
__all__ = ['Agent', 'TabularAgent', 'MonteCarloAgent', 'MonteCarloOffPolicyAgent', 'MonteCarloOnPolicyAgent', 'SARSAAgent', 'QLearningAgent', 'SARSASemiGradientAgent', 'DeepQAgent']

