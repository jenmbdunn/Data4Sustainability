#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

# imports
from pyomo.environ import ConcreteModel, assert_optimal_termination, value
from idaes.core import FlowsheetBlock
from pyomo.util.check_units import assert_units_consistent
from watertap.unit_models.crystallizer_basecode_Dhe import Crystallization
import watertap.property_models.cryst_prop_pack as props
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting, CrystallizerCostType
from idaes.core.util.model_statistics import degrees_of_freedom

import pandas as pd

if __name__ == "__main__":
    # get solver
    solver = get_solver()

    # setup model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.NaClParameterBlock()
    m.fs.costing = WaterTAPCosting()

    # create uit models
    m.fs.unit = Crystallization(property_package=m.fs.properties)
    m.fs.unit.crystal_growth_rate.fix()
    m.fs.unit.souders_brown_constant.fix()
    m.fs.unit.crystal_median_length.fix()
    m.fs.unit.crystallization_yield["NaCl"].fix(0.9) #vary according to desired solid salt yield

    # Move height upper bound
    m.fs.unit.height_crystallizer.setub(50)

    # specify the flowsheet
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(21.133) #vary based on LSRRO brine
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(106.786) #vary based on LSRRO brine
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(1e-6)
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Vap", "H2O"].fix(1e-6)

    m.fs.unit.inlet.pressure[0].fix(101325)
    m.fs.unit.inlet.temperature[0].fix(273.15 + 20)
    m.fs.unit.temperature_operating.fix(273.15 + 55)

    # scaling -- may require changing based on changes to flow and/or yield
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-3, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-1, index=("Liq", "NaCl")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-2, index=("Vap", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-2, index=("Sol", "NaCl")
    )

    iscale.calculate_scaling_factors(m.fs)

    # initialize units
    m.fs.unit.initialize()

    # solve
    results = solver.solve(m, tee=True)

    # costing
    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"cost_type": CrystallizerCostType.mass_basis},
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.unit.properties_in[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.unit.properties_in[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.unit.properties_in[0].flow_vol)

    # consistent units
    assert_units_consistent(m)

    # re-solve model
    results = solver.solve(m, tee=True)
    assert_optimal_termination(results)

    #print(results)
    m.fs.unit.report()
    m.fs.costing.total_capital_cost.display()
    m.fs.costing.total_operating_cost.display()
    m.fs.costing.LCOW.display()
    m.fs.costing.specific_energy_consumption.display()
    m.fs.unit.properties_out[0].flow_vol.display()
    m.fs.costing.aggregate_flow_costs.display()

