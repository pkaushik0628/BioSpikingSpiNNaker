from typing import Iterable
from numpy import floating
from numpy.typing import NDArray

#Spinnaker Stuff
from spinn_utilities.overrides import overrides
from spinn_front_end_common.interfaces.ds import DataSpecificationBase
from spinn_front_end_common.utilities.constants import (BYTES_PER_SORT, BYTES_PER_WORD)
from spynnaker.pyNN.data import SpynnakerDataView
from spynnaker.pyNN.models.neuron.plasticity.stdp.common import (get_exp_lut_array)
from spynnaker.pyNN.models.neuron.plasticity.stdp.timing_dependence import (AbstractTimingDependence)
from spynnaker.pyNN.models.neuron.plasticity.stdp.synapse_structure import (SynapseStructureWeightOnly)

class TimingDependenceBioSpikingTriplet(AbstractTimingDependence):
    __slots__ = ( #Create fixed memory parameter allocation within each created object instance
        "__tau_minus",
        "__tau_minus_data",
        "__tau_plus",
        "__tau_plus_data",
        "__tau_y",
        "__tau_y_data",
        "__a_plus",
        "__a_minus")
    __PARAM_NAMES = ('tau_plus', 'tau_minus', 'tau_y') #tau_plus = pre_fast, tau_minus = post_fast, tau_y = post_slow

    def __init__(self, tau_plus: float, tau_minus: float, tau_y: float, A_plus: float, A_minus: float, learning_rate: float):
        """
        Initialization constructor for the class TimingDependenceBioSpikingTriplet
        tau_plus = pre-trace (fast)
        tau_minus = post-trace (fast)
        tau_y = post_trace(slow)
        a_plus = potentiation amplitude
        a_minus = depression amplitude
        """
        super().__init__(SynapseStructureWeightOnly())
        self.__tau_plus = tau_plus
        self.__tau_minus = tau_minus
        self.__tau_y = tau_y
        self.__a_plus = A_plus
        self.__a_minus = A_minus
        ts = SpynnakerDataView.get_simulation_time_step_ms() #get the timestep/timechunk
        self.__tau_plus_data = get_exp_lut_array(ts, self.__tau_plus)
        self.__tau_minus_data = get_exp_lut_array(ts, self.__tau_minus)
        self.__tau_y_data = get_exp_lut_array(ts, self.__tau_y)
        self.__learning_rate = learning_rate

    """
    Getter Functions
    """

    @property
    def tau_plus(self) -> float:
        """
        Returns fast pre-trace deacy constant
        """
        return self.__tau_plus

    @property
    def tau_minus(self) -> float:
        """
        Returns fast post trace decay constant
        """
        return self.__tau_minus

    @property
    def tau_y(self) -> float:
        """
        Returns slow post trace decay constant
        """
        return self.__tau_y

    @property
    def A_plus(self) -> float:
        """
        Returns potentiation amplitude
        """
        return self.__a_plus

    @property
    def A_minus(self) -> float:
        """
        Returns depression amplitude
        """
        return self.__a_minus


    """
    Setter Functions
    """

    @A_plus.setter 
    def A_plus(self, new_value: float) -> None:
        """
        Sets potentiation amplitude
        """
        self.__a_plus = new_value

    @A_minus.setter
    def A_minus(self, new_value: float) -> None:
        """
        Sets depression amplitude
        """
        self.__a_minus = new_value

    """
    Other functions
    """

    @overrides(AbstractTimingDependence.is_same_as)
    def is_same_as(self, timing_dependence: AbstractTimingDependence) -> bool:
        """
        Check instance for TimingDependenceBioSpikingTriplet
        """
        if not isinstance(timing_dependence, TimingDependenceBioSpikingTriplet):
            return False
        return (
            (self.__tau_plus == timing_dependence.tau_plus) and
            (self.__tau_minus == timing_dependence.tau_minus) and
            (self.__tau_y == timing_dependence.tau_y)
        )

    @property
    def vertex_executable_suffix(self) -> str:
        """
        The suffix to be appended to the vertex executable for this rule.
        """
        return "bio_spiking_triplet"

    @property
    def pre_trace_n_bytes(self) -> int:
        """
        The number of bytes used by the pre-trace of the rule per neuron.
        """
        # Triplet rule trace entries consists of two 16-bit traces - R1 and R2
        # (Note: this is the pre-trace size, not the post-trace size)
        return BYTES_PER_SHORT * 2

    @overrides(AbstractTimingDependence.get_parameters_sdram_usage_in_bytes)
    def get_parameters_sdram_usage_in_bytes(self) -> int:
        """
        Length of look up table data for all exponential decay traces
        """
        lut_array_words = (
            len(self.__tau_plus_data) + len(self.__tau_minus_data) + len(self.__tau_y_data))
        return lut_array_words * BYTES_PER_WORD

    @property
    def n_weight_terms(self) -> int:
        """
        The number of weight terms expected by this timing rule.
        """
        return 1

    @overrides(AbstractTimingDependence.write_parameters)
    def write_parameters(self, spec: DataSpecificationBase, global_weight_scale: float, synapse_weight_scales: NDArray[floating]) -> None:
        # Write lookup tables
        spec.write_array(self.__tau_plus_data)
        spec.write_array(self.__tau_minus_data)
        spec.write_array(self.__tau_y_data)

    @overrides(AbstractTimingDependence.get_parameter_names)
    def get_parameter_names(self) -> Iterable[str]:
        """
        Return parameter names as an iterable list of strings : ('tau_plus', 'tau_minus', 'tau_y')
        """
        return self.__PARAM_NAMES

    

