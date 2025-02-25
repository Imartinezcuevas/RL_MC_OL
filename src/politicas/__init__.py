"""
Module: politicas/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete politicas.

Author: Iván Martínez Cuevas
Email: ivan.martinezc@um.es
Date: 2025/02/25

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# Importación de módulos o clases
from .policy import Policy
from .epsilon_greedy import EpsilonGreedyPolicy
# Lista de módulos o clases públicas

__all__ = ['Policy', 'EpsilonGreedyPolicy']

